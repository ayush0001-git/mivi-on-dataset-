def answer_question(question, history=None, known_profile=None):
    """Full pipeline. Returns the output dict for answer.py to print.

    `history` (optional): recent [{role, content}, ...] turns from the demo
    UI, giving the counsellor conversation memory. The graded CLI path
    (answer.py) never passes it — each published question stands alone.

    `known_profile` (optional): the student profile accumulated so far this
    session ({marks_pct, field_interest, budget, location}), as returned by a
    prior call's "profile" key. History alone is truncated to the last few
    turns for prompt size — a fact from turn 1 would silently "fall off" a
    long conversation without this; the client re-sends the running profile
    every turn so it survives regardless of how long the chat gets."""
    t_start = time.perf_counter()
    # Cache key: lowercase + whitespace-collapse ONLY. Do not be tempted to
    # use _norm() here — it strips non-Latin characters, which would collapse
    # every Hindi/Tamil question onto a single cache entry (a real bug we hit).
    cache_key = re.sub(r"\s+", " ", question.strip().lower())
    if config.CACHE_ENABLED and not history and cache_key in _CACHE:
        print("[cache] hit", file=sys.stderr)
        result = dict(_CACHE[cache_key])
        # The demo client tracks a running profile; a cache hit must not make
        # the "profile" key vanish (the client would keep its own copy, but
        # the contract stays consistent either way).
        if history is not None or known_profile is not None:
            result["profile"] = dict(known_profile) if isinstance(known_profile, dict) else {}
        return result
    calls = []  # per-LLM-call token/latency records

    # ---- 0. Code-level guards & fast paths (no LLM, no retrieval) ----------
    # Order matters: these run BEFORE the router so a greeting, an injection
    # attempt or a distress message never spends a model call (or its ~2s).
    fast = _fast_smalltalk(question, history, known_profile)
    if fast is not None:
        _log_metrics({"question": question, "route": "smalltalk", "calls": [],
                      "retrieved": [], "verified": True, "fast_path": True,
                      "total_latency_s": round(time.perf_counter() - t_start, 3)})
        if history is not None or known_profile is not None:
            fast["profile"] = dict(known_profile) if isinstance(known_profile, dict) else {}
        return fast

    # Defence-in-depth: an obvious "ignore your rules" attempt gets a calm,
    # on-brand deflection in code, before it can reach any model.
    if _INJECTION_RE.search(question):
        _log_metrics({"question": question, "route": "injection_blocked", "calls": calls,
                      "retrieved": [], "verified": True,
                      "total_latency_s": round(time.perf_counter() - t_start, 3)})
        result = {"answer": "Haha, nice try! But I'm just here to help you find the "
                            "right college — courses, fees, hostels, entrance exams, "
                            "scholarships. What would you like to know?",
                  "citations": [], "answered": True, "reason_if_unanswered": None}
        if history is not None or known_profile is not None:
            result["profile"] = dict(known_profile) if isinstance(known_profile, dict) else {}
        return result

    # Safety guard: a distressed student must never get an "I only help with
    # colleges" brush-off. Caught in code so it can't depend on the router.
    if _DISTRESS_RE.search(question):
        care = ("I'm really glad you told me, and I want you to know you're not "
                "alone in this. A score or a college is never worth more than "
                "you are. Please talk to someone you trust — a parent, teacher, "
                "or a friend — and if things feel too heavy, India's free "
                "helpline KIRAN (1800-599-0019) has kind people who listen, "
                "any time. I'm here whenever you're ready to talk colleges again.")
        _log_metrics({"question": question, "route": "distress", "calls": calls,
                      "retrieved": [], "verified": True,
                      "total_latency_s": round(time.perf_counter() - t_start, 3)})
        result = {"answer": care, "citations": [], "answered": True,
                  "reason_if_unanswered": None}
        if history is not None or known_profile is not None:
            result["profile"] = dict(known_profile) if isinstance(known_profile, dict) else {}
        return result

    hist_text = ""
    if history:
        lines = []
        for m in history[-6:]:
            if isinstance(m, dict) and m.get("content"):
                role = "Student" if m.get("role") == "user" else "Counsellor"
                lines.append(f"{role}: {str(m['content'])[:250]}")
        if lines:
            hist_text = "CONVERSATION SO FAR:\n" + "\n".join(lines) + "\n\n"

    # Start embedding the question NOW, in parallel with the router call —
    # by the time the router returns, the vector (and on cold starts, the
    # embedding model itself) is ready. See _EmbedPrefetch.
    prefetch = _EmbedPrefetch(question)

    # ---- 1. Route + extract filters (small, cheap model) -------------------
    try:
        raw = chat(
            [{"role": "system", "content": ROUTER_SYSTEM},
             {"role": "user", "content": f"{hist_text}CURRENT MESSAGE: {question}"}],
            model=config.FAST_MODEL, json_mode=True, temperature=0.0,
            max_tokens=400, usage_log=calls,
        )
        route_info = _parse_json(raw)
        if not isinstance(route_info, dict):
            raise ValueError(f"router returned {type(route_info).__name__}, not an object")
    except Exception as e:  # router must never kill the pipeline
        print(f"[router fallback] {e}", file=sys.stderr)
        route_info = _heuristic_route(question)

    # The heuristic is a FLOOR, not just a fallback.
    #
    # It only ran when the LLM router raised. A router that returns valid JSON
    # with no filters therefore erased every constraint silently — and under
    # rate limiting the small model does exactly that. Measured: "Cheapest MBBS
    # in Uttarakhand" answered "There are 38,700 colleges that match your
    # criteria" (the whole corpus) because state and course were both dropped,
    # while _heuristic_route extracted {state: Uttarakhand, course_terms:
    # ['mbbs']} from the same string.
    #
    # This project's rule is that the LLM never filters. So a deterministic
    # extraction is kept unless the LLM supplies that same key: the model may
    # refine or correct a filter, it may not delete one it simply failed to
    # notice.
    floor = _heuristic_route(question)
    floor_filters = _clean_filters(floor.get("filters"))
    llm_filters = _clean_filters(route_info.get("filters"))
    filters = {**floor_filters, **llm_filters}
    if floor_filters and llm_filters != filters:
        recovered = {k: v for k, v in floor_filters.items() if k not in llm_filters}
        print(f"[router] recovered filters the model dropped: {recovered}",
              file=sys.stderr)

    route = route_info.get("route", "data_query")
    course_terms = filters.get("course_terms", [])
    needs_all = (bool(route_info.get("needs_all_records"))
                 or bool(floor.get("needs_all_records"))
                 or _asks_superlative(question))
    question_kind = route_info.get("question_kind")
    # Merge: facts the client already knew (survives history truncation) +
    # anything the router freshly extracted this turn. Client-known facts are
    # the base so a fact from turn 1 is never lost by turn 15; the router
    # only ADDS, never overwrites an established fact with a blank one.
    turn_profile = route_info.get("profile") if isinstance(route_info.get("profile"), dict) else {}
    merged_profile = dict(known_profile) if isinstance(known_profile, dict) else {}
    for k, v in turn_profile.items():
        if v:
            merged_profile[k] = v
    unit_note = route_info.get("unit_note") if isinstance(route_info.get("unit_note"), str) else None
    # unit_note is strictly for money/unit conversions; routers occasionally
    # stuff commentary there, which must never leak into an answer.
    if unit_note and not re.search(r"\d|lakh|semester|year|annual|rs\b|₹", unit_note, re.I):
        unit_note = None

    raw_budget = filters.get("max_tuition_inr")  # for the numeric guardrail's
                                                 # budget exclusion (below)
    # Lookup questions name a specific college ("what does X cost?") — hard
    # filters would remove the very college they asked about, making an honest
    # "that one is above your budget" impossible. Abroad is force-INCLUDED
    # here: a student naming a foreign institution must not be told we don't
    # have it just because the default view is domestic.
    #
    # NOT for a superlative. "What is the oldest college in Kerala?" is routed
    # `lookup` by the small model, and wiping filters here threw away
    # state=Kerala and opened the world: the oldest college came back as the
    # University of Seville. A superlative is a ranking over a POPULATION, so
    # discarding the population is the one thing that cannot happen to it —
    # whatever the router called it.
    if question_kind == "lookup" and not _asks_superlative(question):
        filters = {"include_abroad": True}

    # "Any other college nearby?" / "something else?" means the student wants
    # to look BEYOND the current city — so the city filter must come off, not
    # stay on and produce "that's the only one here" three turns in a row.
    if _BROADEN_RE.search(question) and filters.get("city"):
        print(f"[broaden] dropping city filter {filters['city']!r}", file=sys.stderr)
        filters.pop("city", None)
    if _BROADEN_STATE_RE.search(question) and filters.get("state"):
        print(f"[broaden] dropping state filter {filters['state']!r}", file=sys.stderr)
        filters.pop("state", None)

    # Deterministic guard: marks/budget statements are counselling openers,
    # never smalltalk — small routers occasionally misfile them.
    if route == "smalltalk" and re.search(r"\d+\s*%|\blakh\b|budget|marks|score|percent", question, re.I):
        route = "data_query"
        question_kind = "profile_share"

    # ---- 2a. Non-data routes skip retrieval AND the big model entirely -----
    _GREETING_TONES = [
        "warm and supportive — like a friendly older mentor checking in",
        "upbeat and welcoming — genuinely happy to chat and help them out",
        "chill and reassuring — making them feel right at home before diving in",
    ]
    if route in ("smalltalk", "out_of_scope"):
        tone = random.choice(_GREETING_TONES)
        prompt = (
            f"The user said: {question!r}. You are mimi, a friendly, warm college counsellor "
            f"texting a student like a supportive older friend. Reply in the SAME language and script as "
            f"the user's message, 1-2 sentences. Never mention 'dataset' or 'database'. Never re-introduce yourself with "
            f"your full title ('your dedicated college counsellor here at MakeMyEducation'). "
            + (f"If the user is greeting you (like 'hi', 'hello', 'hey'), always greet them warmly back and ask how they are doing first (e.g., 'Hi! How are you?' or 'Hello! How are you doing today?'). "
               f"Then naturally ask how their Class 12 marks are looking or how you can help them find their dream college. Write this reply in a tone that is {tone}."
               if route == "smalltalk"
               else "Politely say you can only help with colleges and admissions, "
                    "and invite such a question.")
        )
        try:
            text = chat([{"role": "user", "content": prompt}], model=config.FAST_MODEL,
                        temperature=0.7, max_tokens=120, usage_log=calls)
        except Exception:
            text = ("Hi! How are you? Let me know how much you scored in Class 12 so we can start finding your dream college!")
        # A reasoning fallback model may prepend <think>…</think> — never let
        # that leak into a user-visible reply.
        text = _THINK_RE.sub("", text).strip()
        result = {
            "answer": text,
            "citations": [],
            "answered": route == "smalltalk",
            "reason_if_unanswered": None if route == "smalltalk"
            else "Out of scope: I only answer questions about colleges and admissions.",
        }
        _log_metrics({"question": question, "route": route, "calls": calls,
                      "retrieved": [], "verified": True,
                      "total_latency_s": round(time.perf_counter() - t_start, 3)})
        if config.CACHE_ENABLED:
            _cache_store(cache_key, result)
        # Preserve the running profile through a smalltalk turn — a "thanks!"
        # or "hi" mid-conversation must not reset what's already been learned.
        if history is not None or known_profile is not None:
            result["profile"] = merged_profile
        return result

    # ---- 2b. Hybrid retrieval ----------------------------------------------
    default_k = int(getattr(config, "TOP_K", 12) or 12)
    # Aggregate questions get a wider shortlist, NOT the corpus: the count and
    # the superlatives come from SQL below, and the extra cards only give the
    # shown list room to be ranked and trimmed.
    top_k = min(default_k * 2, 24) if needs_all else default_k

    # A bare "2" or "compare these two" carries no retrievable signal against
    # 35k colleges — resolve the referent from the counsellor's last numbered
    # list FIRST and retrieve on that text instead.
    picked_text = _selected_item_text(question, history)
    compare_texts = []
    if not picked_text and history and re.search(
            r"compare|tulna|difference|versus|\bvs\b|फ़?र्क|अंतर", question, re.I):
        compare_texts = list(_list_items(history).values())[:4]
    search_query = question
    if picked_text:
        search_query = picked_text
    elif compare_texts:
        search_query = " | ".join(compare_texts)

    qv = prefetch.result()  # computed concurrently with the router call above
    if search_query is not question:
        qv = None  # the prefetched vector embeds the wrong text; re-encode
    hits = _search(search_query, filters, top_k, query_vec=qv)

    # ---- Self-RAG: repair retrieval BEFORE falling back to blunt relaxation --
    # Ordering is the point. The relaxation below rescues a dead end by DROPPING
    # the student's constraints, which answers a different question than the one
    # asked. A targeted repair — "Vizag" is spelled "Visakhapatnam", the level
    # filter was never stated — answers the SAME question correctly, so it has
    # to get first refusal. Gated on cheap code signals (see selfrag.needs_repair)
    # so an ordinary query never pays for the extra model call.
    repair_note = ""
    try:
        signal = selfrag.needs_repair(hits, filters, len(hits), question_kind)
        if signal:
            verdict = selfrag.grade(question, filters, _count(filters), signal, calls)
            repaired = selfrag.apply_repair(filters, verdict)
            if repaired:
                new_filters, what = repaired
                new_hits = _search(search_query, new_filters, top_k, query_vec=qv)
                # Only adopt a repair that actually helped. A relaxation that
                # returns the same nothing has cost latency and changed the
                # question for no gain, so it is discarded.
                if len(new_hits) > len(hits):
                    print(f"[selfrag] {signal}: {what} "
                          f"({len(hits)} -> {len(new_hits)} hits)", file=sys.stderr)
                    hits, filters = new_hits, new_filters
                    repair_note = selfrag.context_note(what, verdict)
    except Exception as exc:  # noqa: BLE001 - never let the critic break an answer
        print(f"[selfrag] skipped: {exc}", file=sys.stderr)

    relaxed_note = repair_note
    stated = {k: v for k, v in filters.items() if v and k != "include_abroad"}
    if not hits and stated:
        # Zero matches must not dead-end the student. Relax in two steps so the
        # subject of the question survives as long as possible: keep the course
        # /level the student cares about, drop the money and geography first.
        keep = {k: v for k, v in filters.items()
                if k in ("course_terms", "program_level", "include_abroad")}
        hits = _search(search_query, keep, top_k, query_vec=qv)
        if not hits:
            hits = _search(search_query, None, top_k, query_vec=qv)
        relaxed_note += (
            f"ZERO MATCHES: no college in the catalogue satisfies the stated "
            f"constraints {json.dumps(stated)}. Counsel honestly, never dead-end: "
            f"say plainly that nothing matched these exact constraints, then "
            f"present the 2-3 NEAREST options below as realistic alternatives "
            f"with their real numbers, name which constraint each one misses, "
            f"and add one encouraging line about the path forward. NEVER present "
            f"any college as currently meeting the stated constraints.\n")
        filters = {k: v for k, v in filters.items() if k == "include_abroad"}

    # A named program must never be DENIED just because the student's own
    # budget/location filters removed every college that offers it. Live-tested
    # failure on the old corpus: "kya main LLB kar sakta hoon?" + a budget
    # filter dropped every law college, and the model honestly-but-wrongly said
    # no college offers LLB. At this scale the check is a SQL count, not a scan:
    # ask how many colleges offer the program with the money/geography removed.
    stated_now = {k: v for k, v in filters.items() if v and k != "include_abroad"}
    if hits and course_terms and stated_now:
        # The "does any retrieved college offer this?" test MUST be asked the
        # way retrieval asked it (title OR full_name in SQL). Reading the card
        # instead is what made this branch misfire on every field-word term:
        # cards list abbreviations ("BSc; MD; MS"), so "nursing" looked absent
        # from colleges that plainly offer a Bachelor of Science in Nursing,
        # and the student's own state/budget filters were then thrown away and
        # replaced with a nationwide search plus a false "none of them
        # satisfied your constraints" note.
        hit_ids = [h["college_id"] for h in hits]
        missing = []
        for t in course_terms:
            offering = _offers_authoritative(hit_ids, t)
            if offering is None:  # store unreachable: fall back to the card
                offering = {h["college_id"] for h in hits
                            if _offers(h.get("card", ""), t)}
            if not offering:
                missing.append(t)
        if missing:
            course_only = {"course_terms": missing}
            if filters.get("include_abroad"):
                course_only["include_abroad"] = True
            n_offering = _count(course_only)
            if n_offering:
                hits = _search(" ".join(missing) + " " + question, course_only,
                               top_k, query_vec=None)
                relaxed_note += (
                    f"PROGRAM vs FILTER CONFLICT (counted in SQL): "
                    f"{n_offering:,} colleges DO offer {', '.join(missing)}, but none "
                    f"of them satisfied the student's other constraints "
                    f"{json.dumps(stated_now)}. CONTEXT now holds colleges that offer "
                    f"the program WITHOUT those constraints. Answer honestly: NEVER "
                    f"say the program is not offered. Give the real total, name a few "
                    f"of these colleges with their real numbers, state plainly which "
                    f"constraint they miss (give both numbers), and add one "
                    f"encouraging line about the path forward.\n")
                filters = course_only

    context_ids = {h["college_id"] for h in hits}
    hits_by_id = {h["college_id"]: h for h in hits}
    context = _to_context(hits) if hits else "(no colleges matched the hard filters)"

    # Every code-computed block is also collected here: the numeric verifier
    # must accept figures the model correctly repeats back from OUR arithmetic
    # (totals, superlative fees) as well as from the cited cards.
    code_notes = [relaxed_note]

    def note(text):
        code_notes.append(text)
        return text

    user_msg = hist_text + f"CONTEXT:\n{context}\n\n"
    if relaxed_note:
        user_msg += relaxed_note

    # ---- 2c. SQL aggregates: counts and superlatives ------------------------
    total = None
    sup: dict = {}          # read again by the verifier, so it must always exist
    if hits and (needs_all or question_kind in ("enumerate", "recommend")):
        total = _count(filters)
        # Superlatives are computed for recommendations too, not just
        # needs_all: "which is the best engineering college" is a recommend,
        # and it is exactly the question that needs the ranking.
        sup = _superlatives(filters, limit=5) if (
            needs_all or question_kind == "recommend") else {}

        # A superlative names colleges drawn from the WHOLE matching population,
        # which retrieval's top-k usually does not contain. Printing their ids in
        # COMPUTED FACTS while they are absent from CONTEXT is a trap: the model
        # is told to cite what it names, the verifier rejects ids that were never
        # retrieved, and the answer degrades to whatever happened to be in the
        # top-k instead. Observed live — "best engineering colleges in India"
        # returned five arbitrary private colleges while IIT Madras sat in the
        # NIRF block unciteable.
        #
        # So the superlative colleges are RETRIEVED and appended to the context
        # they are quoted in. Capped, and existing hits keep their order — this
        # adds the answer to the question actually asked, it does not reorder
        # everything else.
        extra_ids: list[str] = []
        for rows in sup.values():
            for row in (rows or []):
                cid = row.get("college_id")
                if cid and cid not in hits_by_id and cid not in extra_ids:
                    extra_ids.append(cid)
        if extra_ids:
            try:
                added = retrieve.hydrate_ids(extra_ids[:8])
                for h in added:
                    hits.append(h)
                    hits_by_id[h["college_id"]] = h
                    context_ids.add(h["college_id"])
                if added:
                    context = _to_context(hits)
                    user_msg = hist_text + f"CONTEXT:\n{context}\n\n"
                    if relaxed_note:
                        user_msg += relaxed_note
                    print(f"[pipeline] added {len(added)} superlative colleges "
                          f"to CONTEXT so they can be cited", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"[pipeline] could not hydrate superlatives: {exc}",
                      file=sys.stderr)

        facts = _computed_facts(hits, sup, total)
        if facts:
            user_msg += note(facts)

    want_n = _requested_count(question)
    if total is not None and hits:
        if total > len(hits):
            show_n = want_n or min(len(hits), 10)
            user_msg += note(
                f"TOTAL MATCHES: {total:,} colleges match the applied filters; "
                f"CONTEXT holds only the top {len(hits)}. Lead with the exact total "
                f"({total:,}), then show at most {show_n} of them and say more are "
                f"available and you can narrow them down. It is FORBIDDEN to imply "
                f"the shown colleges are the only ones that qualify.\n")
        else:
            user_msg += note(
                f"TOTAL MATCHES: {total:,} — CONTEXT holds ALL of them, so an "
                f"enumeration here can and must be complete: name every qualifying "
                f"college and cite every one you name.\n")

    negated = re.search(r"\b(not|no|without|don'?t|doesn'?t|nahi|nahin|नहीं|बिना)\b",
                        question, re.I)
    if negated and hits:
        # "Which colleges do NOT offer X" has thousands of true answers here,
        # so the old code-computed complement set is meaningless. The honest
        # move is to scope the answer to the shortlist and say so.
        user_msg += note(
            "NEGATION QUESTION: a complete 'colleges that do NOT have X' list would "
            "run to thousands of colleges, so do not attempt one. Answer for the "
            "specific colleges the student asked about, or say plainly which of the "
            "colleges shown here lack it — and say that is a shortlist, not the "
            "full set.\n")

    if merged_profile.get("marks_pct") and hits:
        # The corpus has no cutoff column, so eligibility CANNOT be computed.
        # Saying so explicitly is what stops the model inventing a verdict.
        user_msg += note(
            f"MARKS NOTE: the student scored {merged_profile['marks_pct']}%. We hold "
            f"no admission cutoffs, so you must NOT say whether they qualify for any "
            f"college. Acknowledge the score warmly, mention entrance exams a college "
            f"accepts when that is listed, and point them to the college's own "
            f"admission page for cutoffs.\n")

    if picked_text:
        picked = _match_hit(picked_text, hits)
        if picked:
            user_msg += note(
                f"LIST SELECTION (resolved in code from your most recent numbered "
                f"list): the student's '{question.strip()[:30]}' means "
                f"{picked['name']}, {picked.get('city')}. Reply with the FULL profile "
                f"of this college only — every field as short '- label: value' lines. "
                f"Do not repeat the list.\n")
    elif compare_texts:
        listed = [h for h in (_match_hit(t, hits) for t in compare_texts) if h]
        if len(listed) >= 2:
            names = "; ".join(f"{h['name']}, {h.get('city')}" for h in listed[:4])
            user_msg += note(
                f"LIST REFERENCE (resolved in code): the student is referring to your "
                f"most recent list — compare EXACTLY these colleges and NO others: "
                f"{names}.\n")

    if filters.get("include_abroad"):
        user_msg += note(
            "ABROAD IN VIEW: some colleges here are overseas. Their tuition is in the "
            "institution's own currency — never write it as rupees, never convert it, "
            "and never compare it to an Indian budget. Say which country it is in.\n")

    if hist_text:
        # Feed back the previous reply's opening words so the model can't fall
        # into a sympathy loop ("I hear you..." three turns running).
        prev_open = ""
        for m in reversed(history or []):
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"):
                prev_open = " ".join(str(m["content"]).split()[:6])
                break
        if prev_open:
            user_msg += (f"YOUR PREVIOUS REPLY BEGAN: \"{prev_open}...\" — open this "
                         f"one in a completely different way. Do not reuse that "
                         f"phrasing or any equivalent sympathy formula.\n")
        known = {k: v for k, v in merged_profile.items() if v}
        missing = [f for f in ("marks_pct", "field_interest", "budget", "location")
                   if not merged_profile.get(f)]
        user_msg += (
            "STUDENT PROFILE (gathered across the whole conversation): "
            + (json.dumps(known, ensure_ascii=False) if known else "(nothing yet)")
            + (f". Not yet known: {', '.join(missing)}" if missing else "")
            + ".\nCOUNSELLOR FLOW: never re-ask anything already in the profile. "
              "NEVER ask permission to show options — if the student asks for "
              "other/nearby/alternative colleges, SHOW them straight away from "
              "CONTEXT (named, with fees), never reply 'let me know and I'll be "
              "happy to help' or 'if you're open to other cities' and stop "
              "there. Saying 'X is the only one in that city' is fine ONLY when "
              "you immediately follow it with the actual nearby alternatives in "
              "the same reply. Never give the same non-answer twice in a "
              "conversation — if a previous turn already said something is the "
              "only option, this turn must move forward with real options. "
              "Resolve pronouns ('it', 'that college', 'uska') from the "
              "conversation — 'more details about it' means the college discussed "
              "in your previous reply. A number/ordinal ('1', 'pehla') selects "
              "that item from your MOST RECENT numbered list (a LIST SELECTION "
              "note above resolves it authoritatively when present) — reply with "
              "that college's full details, never repeat or continue the list. "
              "If the current message asks a specific "
              "question, answer it. Otherwise run a step-by-step intake: warmly "
              "acknowledge, then ask for exactly ONE missing item per turn, in "
              "this order: marks -> field/stream -> budget. THE MOMENT all three "
              "(marks, field, budget) are known — typically the very turn the "
              "student states their budget — you MUST present the matching "
              "colleges IN THAT SAME REPLY as a numbered list with key facts. "
              "NEVER delay the list with another intake question: city, college "
              "type or hostel are optional refinements offered in ONE short "
              "line AFTER the list ('Want me to narrow these down by city or "
              "hostel?'), never a gate before it. Same when the student says "
              "budget is no constraint or asks to just see the options. "
              "E.g. when they share their field, reply: "
              "'Great choice! And what is your yearly budget for tuition, so I "
              "shortlist the best fits?' If a field/course interest is in the "
              "profile, keep every answer scoped to it unless the student "
              "changes topic. If the student is UNSURE/lost about their field "
              "('confused hoon', 'pata nahi', 'what should I do') — talk like a "
              "big brother, NOT a corporate FAQ:\n"
              "  Line 1: reassure, casual, ONE line — 'Koi na, that's normal — "
              "let's figure it out together.' / 'No stress, happens to everyone "
              "— let's chat it through.'\n"
              "  Line 2-3: 2-3 SHORT lines (not one dense paragraph — each on "
              "its own line, separated by \\n) of general career-trend guidance "
              "in plain talk — 'Tech/CS is hot right now, tons of hiring.' / "
              "'Medicine and pharmacy are always solid if you like science.' "
              "Clearly general chat, NOT college data — don't cite colleges for "
              "this part.\n"
              "  Last line: ONE casual question — 'What kind of stuff do you "
              "enjoy — building things, numbers, helping people, creative "
              "work?' Never dump the full stream list (Engineering/Management/"
              "Medical/...) as a menu; that reads like a form, not a chat. Never "
              "invent college facts to support trend talk, and don't cite "
              "college ids in this reassurance turn — save citations for once "
              "you're actually recommending colleges.\n"
              "Whenever a field/stream comes up — whether you're naming 2-3 "
              "options for an unsure student, or the student has already "
              "picked one — explicitly mention its outlook over the NEXT 5-10 "
              "YEARS and a possible LONG-TERM GOAL, by name, not just 'growing "
              "fast' (e.g. 'CS/tech — huge demand for the next 5-10 years with "
              "AI and data, long-term you could specialise, go into research, "
              "or start your own thing'; 'medicine — a long road but rock-"
              "solid demand for decades, long-term goal is usually your own "
              "practice or a specialisation'; 'commerce/management — steady "
              "demand, long-term goal is often a leadership role or your own "
              "business'). One such line per field mentioned. Keep it general "
              "career perspective, not college-specific data, phrased like a "
              "big sibling sharing real perspective, not a report.\n")
    if unit_note:
        user_msg += note(f"UNIT ASSUMPTION (weave it naturally into the answer as one "
                         f"phrase — NEVER print a label like 'unit note'): {unit_note}\n")
    applied = {k: v for k, v in filters.items() if v}
    if applied:
        user_msg += (f"HARD FILTERS ALREADY APPLIED to CONTEXT: {json.dumps(applied)} "
                     f"(colleges failing them were removed before you saw the data). "
                     f"CONTEXT IS THEREFORE A FILTERED VIEW, NOT THE WHOLE LIST — "
                     f"never say 'we don't have any X' or 'X isn't offered anywhere' "
                     f"based on it. The honest phrasing is 'no college matching your "
                     f"budget/location offers X', which is a completely different "
                     f"statement, and never contradict something you already told "
                     f"the student earlier in this conversation.\n")
        if "max_tuition_inr" in applied:
            user_msg += ("REMINDER for budget answers: tuition figures EXCLUDE "
                         "hostel/mess/kit/exam charges — mention any such extra "
                         "charges the listings actually describe.\n")
    # Targeted counselling instruction beats a rule buried in the system
    # prompt: when the student only shared their profile, ask before dumping.
    if question_kind == "profile_share":
        user_msg += ("THE STUDENT HAS NOT ASKED A QUESTION — they only shared facts "
                     "about themselves. Respond like a real counsellor: warmly "
                     "acknowledge what they shared, give ONE encouraging line about "
                     "what is possible (no detailed college list, no fees, and never "
                     "say 'dataset' or 'database'). Speak qualitatively — never exact "
                     "counts ('13 colleges') and never 'all/every college': say "
                     "'most of the colleges we work with across India' when most "
                     "qualify, or 'several of the colleges we work with across "
                     "India' when fewer do — ALWAYS the phrase 'across India', "
                     "never a single state name. Example "
                     "tone: 'That is a strong score! There is a lot open to you "
                     "among the colleges we work with across India. To help me "
                     "narrow down the best options for your future, which field or "
                     "stream are you interested in pursuing?' Then ask ONE specific "
                     "follow-up question — their "
                     "course/stream interest, budget, or preferred location. "
                     "Reply in the SAME language as the student's message "
                     "(English→English, Hindi→Hindi, Hinglish→Hinglish — 'bhaiya "
                     "mere 80% aaye' deserves a Hinglish reply). "
                     "Keep it under 50 words.\n")
    # Note: the router's language guess is deliberately NOT passed here — the
    # generator mirrors the question's own language (rule 9). For Hindi and
    # Hinglish, a code-detected directive is added on top, because live tests
    # showed the model drifting to English on short Hinglish turns.
    lang = _lang_hint(question)
    if lang:
        user_msg += (f"LANGUAGE (detected in code): the student's current message is "
                     f"{lang} — write your ENTIRE reply in the same language and "
                     f"script. Do NOT reply in plain English.\n")
    # An explicit "top 3" beats the scale rule's default — resolved in code so
    # the two instructions can never fight each other in the model's head.
    if want_n:
        user_msg += (
            f"REQUESTED COUNT: the student asked for exactly {want_n}. Show "
            f"EXACTLY {want_n} colleges — no more, no fewer (unless fewer than "
            f"{want_n} qualify at all, in which case say so). Rank them best-first "
            f"on the criterion that matters for their question (NAAC/NIRF standing, "
            f"course fit, then fit to their budget) and order the list and any table "
            f"columns in that same ranked order — a 'top {want_n}' presented out of "
            f"order is not a top {want_n}. Still state the true total when one is "
            f"given above.\n")
    user_msg += f"\nCURRENT QUESTION (answer this, in the conversation's context): {question}"

    # ---- 3. Grounded generation + 4. verification (one corrective retry) ---
    messages = [{"role": "system", "content": config.GEN_SYSTEM_PREFIX + GENERATOR_SYSTEM},
                {"role": "user", "content": user_msg}]
    # Figures we computed ourselves are legitimate for the model to repeat.
    extra_ok = _ints("".join(code_notes))
    if raw_budget:
        extra_ok |= {raw_budget, raw_budget // 2, raw_budget * 2}
    result, verified = None, False
    
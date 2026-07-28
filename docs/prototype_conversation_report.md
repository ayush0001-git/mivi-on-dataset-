# Live conversation test report

## S1: English counsellor intake flow

**You:** hi

**mimi (0.0s, PASS):**
```
Hello hello! 😊 I'm mimi — how's it going? Let's find you a great college: how much did you score in Class 12 (%)?
```
citations: [] · answered: True

**You:** i got 68 percent in class 12

**mimi (57.2s, PASS):**
```
That is a solid score! You are eligible for several of the colleges we work with across India. To help me narrow down the best options for your future, which field or stream are you interested in pursuing?
```
citations: ['C014', 'C015', 'C002', 'C013', 'C007', 'C008', 'C010', 'C005'] · answered: True

**You:** i am interested in computer science

**mimi (2.6s, PASS):**
```
That's a great choice! CS is booming with huge demand for the next 5-10 years due to AI and data, and long-term you could specialize, go into research, or even start your own tech venture. To help me find the best fit for you, what is your yearly budget for tuition?
```
citations: [] · answered: True

**You:** my budget is 1.2 lakh per year

**mimi (2.8s, PASS):**
```
That's a very reasonable budget to work with, and it opens up some great paths for you! With your score and interest in Computer Science, here are the colleges that fit your criteria:
1. Ganga Valley University, Haridwar — Rs 98,000/year; offers BCA.
2. Shivalik Government Polytechnic, Roorkee — Rs 22,000/year; offers Diploma CSE.
Note: Shivalik Government Polytechnic awards diplomas, not degrees. Fees are tuition only and exclude hostel, mess, and other extra charges mentioned in the college profiles. Would you like me to narrow these down by city or hostel availability?
```
citations: ['C002', 'C005'] · answered: True

**You:** 2

**mimi (8.6s, PASS):**
```
Shivalik Government Polytechnic is a solid, budget-friendly option for your interest in computer science. Here are the details:
- Type: Government institution
- City: Roorkee
- Courses offered: Diploma CSE; Diploma ME; Diploma Civil
- Tuition: Rs 22,000 per academic year
- Last-year cutoff: 60% aggregate
- Total seats: 200
- Hostel available: No
- NAAC grade: B+
- Average placement: 3.4 LPA average
- Established: 1998
- About: This is a state-run polytechnic offering three-year diploma programmes only; it does not award degrees. Tuition is heavily subsidised and waived entirely for SC/ST students and those with family income below 2.5 lakh per year. Placements are primarily for technician roles, and many students use this as a pathway to enter the second year of a degree programme elsewhere.
```
citations: ['C005'] · answered: True

**You:** compare 1 and 2

**mimi (4.3s, PASS):**
```
| Field | Ganga Valley University | Shivalik Government Polytechnic |
| Type | Private | Government |
| Courses | BBA, B.Com, BCA, MBA | Diploma CSE, Diploma ME, Diploma Civil |
| Tuition/year | Rs 98,000 | Rs 22,000 |
| Cutoff | 68% | 60% |
| Seats | 360 | 200 |
| Hostel | Yes | No |
| NAAC | B++ | B+ |
| Avg placement | 4.2 LPA | 3.4 LPA |
| Scholarships | Need-based for income < 4 lakh | SC/ST waiver, need-based for income < 2.5 lakh |
| Established | 2009 | 1998 |

Ganga Valley University offers a broader range of degree programs and campus facilities, while Shivalik Government Polytechnic focuses on technical diploma training with significantly lower tuition.
The main trade-off is that Ganga Valley University provides a full university experience with hostel facilities, whereas Shivalik Government Polytechnic is a commuter-only institution that requires students to arrange their own transport.
```
citations: ['C002', 'C005'] · answered: True

**You:** thanks!

**mimi (0.0s, PASS):**
```
Anytime! 😊 That's what I'm here for — ask me whenever the next question pops up.
```
citations: [] · answered: True

## S2: Hinglish flow — low score + sarkari + hostel

**You:** bhaiya mere 55% aaye hain, koi sarkari college milega kya?

**mimi (3.3s, PASS):**
```
Arre, tension mat lo, 55% ek solid score hai aur tumhare liye sarkari colleges ke options open hain. Hum jin colleges ke saath kaam karte hain, unme se kuch mein tumhari eligibility banti hai. Yeh batao, tumhara interest kaunse course ya stream mein hai?
```
citations: ['C007'] · answered: True

**You:** hostel bhi chahiye mujhe

**mimi (3.1s, PASS):**
```
Hostel ki tension mat lo, Kumaon Arts and Science College, Almora mein hostel facility available hai. 

1. Kumaon Arts and Science College, Almora — Rs 15,000/year; government institution with 480 seats, NAAC B grade, and 2.8 LPA average placement. 

Note: Fees are tuition only and exclude hostel, mess, and other extra charges. Hostel seats are limited and given to students from remote hill blocks first. Admission is based on Class 12 marks with no entrance exam.
```
citations: ['C007'] · answered: True

**You:** uski fees kitni hai?

**mimi (2.5s, PASS):**
```
Kumaon Arts and Science College, Almora ki tuition fees Rs 15,000 per academic year hai. Yeh fees sirf tuition ke liye hai aur ismein hostel, mess ya koi aur extra charges shamil nahi hain.
```
citations: ['C007'] · answered: True

## S3: Same facts asked different ways (one-shot each)

**You:** Sabse sasta college kaun sa hai?

**mimi (2.5s, PASS):**
```
Sabse sasta college Kumaon Arts and Science College, Almora hai, jahan tuition fees Rs 15,000 per academic year hai. Iske baad Shivalik Government Polytechnic, Roorkee aata hai, jahan tuition fees Rs 22,000 per academic year hai. Note: ye fees sirf tuition ke liye hain aur ismein hostel, mess ya koi aur extra charges shamil nahi hain.
```
citations: ['C007', 'C005'] · answered: True

**You:** Which college has the highest placement AND also a hostel?

**mimi (2.3s, PASS):**
```
Ambedkar National Institute, Kashipur is the college with the highest reported average placement of 8.4 LPA, and it does provide hostel facilities for its students.
```
citations: ['C012'] · answered: True

**You:** How many colleges offer a B.Tech?

**mimi (7.0s, PASS):**
```
There are 4 colleges in the list that offer a B.Tech program:
1. Terai Technical University, Rudrapur — Rs 118,000/year.
2. North Ridge Institute of Technology, Dehradun — Rs 145,000/year.
3. Ambedkar National Institute, Kashipur — Rs 45,000/year.
4. Himalayan College of Engineering, Roorkee — Rs 132,000/year.
Note: fees are tuition only and exclude hostel, mess, and other extra charges.
```
citations: ['C009', 'C001', 'C012', 'C003'] · answered: True

**You:** Does Ganga Valley University offer a PhD in Physics?

**mimi (2.5s, PASS):**
```
We don't have a PhD in Physics listed for Ganga Valley University, Haridwar — its courses with us are BBA, B.Com, BCA, and MBA. If you're interested in research, Terai Technical University in Rudrapur does admit a small number of doctoral scholars in engineering disciplines.
```
citations: ['C002', 'C009'] · answered: False

**You:** I can pay 60 thousand per semester. Which colleges fit my budget?

**mimi (3.5s, PASS):**
```
That's a solid budget to work with — at 60,000 per semester, that's 120,000 per year, which opens up almost all the colleges we work with. Here are the colleges that fit your budget:
1. Ambedkar National Institute, Kashipur — Rs 45,000/year; hostel is compulsory for all undergraduates.
2. Terai Technical University, Rudrapur — Rs 118,000/year; hostel and mess are charged separately.
3. Ganga Valley University, Haridwar — Rs 98,000/year; hostel and mess are charged separately.
4. Char Dham Pharmacy College, Srinagar — Rs 95,000/year; laboratory and dissertation charges in the final year are billed extra.
5. Ganga Institute of Commerce, Dehradun — Rs 72,000/year; no hostel facilities available.
6. Shivalik Government Polytechnic, Roorkee — Rs 22,000/year; no hostel facilities available.
7. Haldwani Law College, Haldwani — Rs 28,000/year; no hostel facilities available.
8. Kumaon Arts and Science College, Almora — Rs 15,000/year; hostel places are limited and allocated to students from remote hill blocks first.

Would you like me to narrow these down by your preferred field of study or location?
```
citations: ['C012', 'C009', 'C002', 'C013', 'C014', 'C005', 'C011', 'C007'] · answered: True

**You:** placement kaisa hai Nainital Institute of Medical Sciences ka?

**mimi (2.9s, PASS):**
```
Nainital Institute of Medical Sciences doesn't have a campus placement figure, but that's completely normal for medical colleges. MBBS and BDS graduates don't go through campus recruitment; instead, they complete a compulsory rotating internship and then move on to postgraduate entrance exams or clinical practice. B.Sc Nursing students, however, are absorbed in large numbers by the attached 600-bed teaching hospital.
```
citations: ['C006'] · answered: True

**You:** Which college has the best NIRF ranking?

**mimi (4.1s, PASS):**
```
I'll check that for you — actually, none of the colleges we work with have an NIRF ranking. If you're looking for the strongest academic standing, Ambedkar National Institute in Kashipur holds an A+ NAAC grade and reports the highest average placement at 8.4 LPA.
```
citations: ['C012'] · answered: False

**You:** kya main LLB kar sakta hoon? mere 70% hain

**mimi (3.0s, PASS):**
```
Dekho, LLB ke liye humare paas sirf ek hi college hai, Haldwani Law College, Haldwani. Par tumhare 70% marks ke saath tum wahan eligible nahi ho paoge kyunki unka cutoff 74% hai. 

Koi na, tension mat lo — tum improvement exam de sakte ho ya phir kisi aur stream mein try kar sakte ho jo tumhare marks ke hisaab se fit ho. Agar tumhara interest law mein hi hai, toh tum agle saal ke liye prepare kar sakte ho, ya phir hum kisi aur field ke baare mein baat karein jo tumhare marks ke liye perfect ho?
```
citations: ['C011'] · answered: True

## Summary
18/18 turns passed · total wall time 157s

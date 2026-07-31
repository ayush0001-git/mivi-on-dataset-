import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test Dataset - complex questions to prevent trivial cache hits
const queries = new SharedArray('queries', function () {
  return [
    "Best MBA colleges in Delhi under 5 lakh",
    "Compare IIT Delhi and IIT Bombay",
    "Best BTech colleges with AI",
    "Show placements in SRM",
    "Which college has lowest fees",
    "Scholarships in VIT",
    "Best colleges for CSE",
    "Top medical colleges",
    "Engineering colleges in Karnataka",
    "What is the fee structure for B.Com in DU",
    "Which is better for law, NLU Delhi or NALSAR?",
    "Cheapest nursing colleges in Kerala",
    "Are there any government pharmacy colleges in Pune?",
    "List of colleges offering BCA with hostel facility",
    "Does BITS Pilani offer scholarships?"
  ];
});

export const options = {
  thresholds: {
    http_req_failed: ['rate<0.01'], // < 1% errors
    http_req_duration: ['p(95)<1500'], // 95% of requests should be below 1500ms
  },
};

export default function () {
  // Pick a random query
  const randomQuery = queries[Math.floor(Math.random() * queries.length)];
  
  // Use a unique session ID for each virtual user to simulate real users
  const sessionId = `test_session_${__VU}`;
  
  const payload = JSON.stringify({
    message: randomQuery,
    session_id: sessionId,
    profile: {} // Empty profile
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: '10s'
  };

  // Hit the chat endpoint
  const res = http.post(`${BASE_URL}/api/chat`, payload, params);
  
  // Basic verifications
  check(res, {
    'status is 200': (r) => r.status === 200,
    'has valid answer': (r) => {
        try {
            const body = r.json();
            return body && (body.answered === true || body.answered === false);
        } catch(e) {
            return false;
        }
    },
  });

  // Short pause to simulate think time between requests
  sleep(1);
}

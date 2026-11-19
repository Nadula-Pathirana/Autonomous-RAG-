# Evaluation run 2026-11-19 19:44

Model: `qwen/qwen3.8-27b` | TOP_K: 3 | MAX_DISTANCE: 0.75

**Passed 25/28 graded** (0 infrastructure errors not counted)
Tool selection 27/28 | Retrieval recall 23/23 | Citation 28/28 | Abstention 1/3 | Faithfulness 26/26 | Relevance 26/26
Avg latency 27.0s | Avg tokens 3086

| Question | Status | Tools | Tool | Retrieval | Content | Citation | Abstain | Faithful | Relevant | Sec | Tokens | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| price-match | PASS | kb | ok | ok | ok | ok | - | ok | ok | 3.7 | 2553 |  |
| price-match-exclusions | PASS | kb | ok | ok | ok | ok | - | ok | ok | 3.2 | 2595 |  |
| return-window | PASS | kb | ok | ok | ok | ok | - | ok | ok | 8.3 | 2455 |  |
| data-security | PASS | kb | ok | ok | ok | ok | - | ok | ok | 22.0 | 2529 |  |
| kb-payment-and-fraud | PASS | kb | ok | ok | ok | ok | - | ok | ok | 28.4 | 2909 |  |
| kb-shipping-address-change | PASS | kb | ok | ok | ok | ok | - | ok | ok | 23.1 | 2507 |  |
| kb-damaged-order-doa | PASS | kb | ok | ok | ok | ok | - | ok | ok | 22.0 | 2450 |  |
| kb-e-waste-recycling | PASS | kb | ok | ok | ok | ok | - | ok | ok | 23.6 | 2614 |  |
| kb-shopify-scopes-and-sync | PASS | kb | ok | ok | ok | ok | - | ok | ok | 25.3 | 2686 |  |
| kb-api-key-rotation | PASS | kb | ok | ok | ok | ok | - | ok | ok | 24.2 | 2598 |  |
| kb-shopify-error-429-422 | PASS | kb | ok | ok | ok | ok | - | ok | ok | 27.9 | 2990 |  |
| kb-warranty-terms | PASS | kb | ok | ok | ok | ok | - | ok | ok | 25.2 | 2627 |  |
| kb-warranty-exclusions | PASS | kb | ok | ok | ok | ok | - | ok | ok | 26.9 | 2891 |  |
| kb-aethercare-deductible | PASS | kb | ok | ok | ok | ok | - | ok | ok | 29.5 | 3085 |  |
| kb-cross-doc-warranty-sync | PASS | kb | ok | ok | ok | ok | - | ok | ok | 32.1 | 3313 |  |
| web-news | PASS | web | ok | ok | ok | ok | - | ok | ok | 46.0 | 4831 |  |
| web-shopify-api-updates | FAIL | kb, web | FAIL | ok | ok | ok | - | ok | ok | 44.2 | 4542 |  |
| web-aes-tls-standards | PASS | web | ok | ok | ok | ok | - | ok | ok | 52.9 | 5174 |  |
| both-price-match-competitors | PASS | kb, web | ok | ok | ok | ok | - | ok | ok | 48.3 | 4527 |  |
| both-warranty-vs-external-gpu | PASS | kb, web | ok | ok | ok | ok | - | ok | ok | 49.8 | 4881 |  |
| both-klarna-financing-overview | PASS | kb, web | ok | ok | ok | ok | - | ok | ok | 55.5 | 5305 |  |
| greeting | PASS | - | ok | - | ok | ok | - | - | - | 5.3 | 1208 |  |
| none-capability-check | PASS | - | ok | - | ok | ok | - | - | - | 6.2 | 1160 |  |
| not-in-documents | PASS | kb | ok | - | ok | ok | ok | ok | ok | 9.2 | 1442 |  |
| not-in-documents-kb104 | FAIL | kb | ok | - | ok | ok | FAIL | ok | ok | 27.4 | 2853 |  |
| not-in-documents-support-hours | FAIL | kb | ok | - | ok | ok | FAIL | ok | ok | 23.5 | 2596 |  |
| edge-vague-kb-first | PASS | kb | ok | ok | ok | ok | - | ok | ok | 40.3 | 4684 |  |
| edge-specific-error-401 | PASS | kb | ok | ok | ok | ok | - | ok | ok | 21.6 | 2415 |  |

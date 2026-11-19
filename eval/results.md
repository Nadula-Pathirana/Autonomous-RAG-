# Evaluation run 2026-11-19 10:30

Model: `qwen/qwen3.8-27b` | TOP_K: 3 | MAX_DISTANCE: 0.75
**Passed 23/28**

| Question | Pass | Tools used | Tool | Content | Citation | Error |
|---|---|---|---|---|---|---|
| price-match | Passed | kb | Passed | Passed | Passed |  |
| price-match-exclusions | Failed | - | Failed | Failed | Failed | Error code: 404 - {'error': {'message': 'The model `minimaxai/minimax-m2.7` does |
| return-window | Passed | kb | Passed | Passed | Passed |  |
| data-security | Passed | kb | Passed | Passed | Passed |  |
| kb-payment-and-fraud | Passed | kb | Passed | Passed | Passed |  |
| kb-shipping-address-change | Passed | kb | Passed | Passed | Passed |  |
| kb-damaged-order-doa | Passed | kb | Passed | Passed | Passed |  |
| kb-e-waste-recycling | Passed | kb | Passed | Passed | Passed |  |
| kb-shopify-scopes-and-sync | Passed | kb | Passed | Passed | Passed |  |
| kb-api-key-rotation | Passed | kb | Passed | Passed | Passed |  |
| kb-shopify-error-429-422 | Passed | kb | Passed | Passed | Passed |  |
| kb-warranty-terms | Passed | kb | Passed | Passed | Passed |  |
| kb-warranty-exclusions | Passed | kb | Passed | Passed | Passed |  |
| kb-aethercare-deductible | Passed | kb | Passed | Passed | Passed |  |
| kb-cross-doc-warranty-sync | Passed | kb | Passed | Passed | Passed |  |
| web-news | Passed | web | Passed | Passed | Passed |  |
| web-shopify-api-updates | Passed | web | Passed | Passed | Passed |  |
| web-aes-tls-standards | Failed | - | Failed | Failed | Failed | Error code: 404 - {'error': {'message': 'The model `minimaxai/minimax-m2.7` does |
| both-price-match-competitors | Passed | kb, web | Passed | Passed | Passed |  |
| both-warranty-vs-external-gpu | Passed | kb, web | Passed | Passed | Passed |  |
| both-klarna-financing-overview | Passed | kb, web | Passed | Passed | Passed |  |
| greeting | Passed | - | Passed | Passed | Passed |  |
| none-capability-check | Passed | - | Passed | Passed | Passed |  |
| not-in-documents | Failed | - | Failed | Passed | Passed |  |
| not-in-documents-kb104 | Failed | - | Failed | Failed | Failed | Error code: 404 - {'error': {'message': 'The model `minimaxai/minimax-m2.7` does |
| not-in-documents-support-hours | Passed | kb | Passed | Passed | Passed |  |
| edge-vague-kb-first | Failed | - | Failed | Passed | Failed |  |
| edge-specific-error-401 | Passed | kb | Passed | Passed | Passed |  |

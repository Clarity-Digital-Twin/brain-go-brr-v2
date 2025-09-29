1. find any print statements, create consistent logging system

2. examine test suite configs and overall test suite to ensure clean and not adhoc patchwork for OOM fixes, or anything esle 

3. LOCAL CACHEBUILDING -- WHATEVER LEADS TO BUILDING THIS IN THE LOCAL PIPELINE TO UPLAOD TO S3 SHOULD CREATE THE PROPER CAHCE MANIFEST JSON THAT MODAL NEEDS TO TRAIN 

4.   - “ResourcesConfig” unused at runtime, but kept intentionally for future use.       - Decide on keeping or removing ResourcesConfig

5. LOOK AT PERFORMANCE STABILITY / OPTIMIZATION?

6. IMPORT AUDIT - 
● ✅ IMPORT AUDIT COMPLETE - WE'RE ACTUALLY CONSISTENT!

  🎯 CURRENT STATE (AFTER AUDIT)

  ✅ src/brain_brr/**/*.py    → 27 files use `from src.brain_brr...` (CORRECT)
  ✅ tests/**/*.py           → 69 files use `from src.brain_brr...` (CORRECT)  
  ✅ deploy/modal/*.py       → 5 functions use `from src.brain_brr...` (CORRECT)
  ✅ __init__.py files       → Use relative imports `from .module` (CORRECT)
  ✅ NO FILES use `from brain_brr...` anywhere (VERIFIED)

  📐 THE RULE WE'RE FOLLOWING

  1. Package internals (__init__.py): Use relative imports from .module
  2. Everything else: Use from src.brain_brr...
  3. Why this works:
    - python -m src train → Works ✅
    - pytest tests/ → Works ✅
    - Modal deployment → Works ✅
    - Package is at src/brain_brr and we run from project root

  🔍 VERIFICATION


7. RACE CONDITIONS????

8. MEMORY LEAKS

9. bogus mocked tests


10. REEPEATED CODE 


11. *** CACHE MANIFEST BUILDING LOCALLY, TRANSFER TO S3, TRANFER TO MODAL. 
    - ONCE AUTO PIPELINE IS SET UP, DELETE LOCAL CACHES AND REBUILD AS ONE WOULD EXPECT TO SEE IF IT ALL WORKS FROM LOCAL SMOKE / TRAINING --> CACHE BUILDING ---> UPLOAD TO S3 ---> TRANSFER TO MODAL
      - WE SHOULD HAVE STEPS THAT FIRST BUILD CACHES / MANIFESTS LOCALLY FOR ALL DATA SETS NEEDED (TRAIN / DEV... DOES EVAL NEED THIS?) ---> THEN COMMANDS THAT TRANSFER LOCAL CACHE / MANIFEST TO S3 --> COMMANDS TO TRANSFER TO MODAL.... AT MINIMUM ----> MAKE SURE THIS IS STILL GOOD AND OKAY
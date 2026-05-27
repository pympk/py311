

poetry run code2prompt --path "C:\Users\ping\Files_win10\python\py311\stocks\notebooks_RLVR_v2" | Out-File -Encoding utf8 codebase.txt

poetry run pytest tests/ -v

### 1. Exclude specific folders (e.g., `venv` or `__pycache__`)
Use the `**/folder_name/**` pattern to ensure it catches the folder regardless of where it is in the directory tree.

```powershell
poetry run code2prompt --path "C:\Users\ping\Files_win10\python\py311\stocks\notebooks_RLVR_v2" --filter "**/*engine.py,**/*features.py,**/tests/test*.py,**/*main.ipynb" --exclude "**/__init__.py" | Out-File -Encoding 
utf8 codebase.txt
```  
---  

✦ I have completed the replacement of all emojis with their corresponding text labels across all  files in the core/ directory.

  ✅ Final Summary of Replacements:
   - Status & Integrity:
     - ❌ ➔ [ERROR]
     - ✅ ➔ [OK]
     - ⚠️ ➔ [WARNING]
     - 🚨 ➔ [CRITICAL]
     - 💥 ➔ [CRASH]
     - ✓ ➔ [OK]
     - ✗ ➔ [FAIL]
   - Audit & Research:
     - 🛡️ ➔ [GUARD]
     - 🔍 ➔ [SEARCH]
     - 🕵️ ➔ [AUDIT]
     - 📝 ➔ [NOTE]
     - 📍 ➔ [INFO]
     - 🏷️ ➔ [LABEL]
     - 💡 ➔ [INFO]
     - 🏗️ ➔ [INFO]
   - Execution & System:
     - ⚡ ➔ [EXEC]
     - 🚀 ➔ [EXEC]
     - 🖥️ ➔ [SYSTEM]
     - 🐛 ➔ [DEBUG]
     - ✨ ➔ [DONE]
     - 💾 ➔ [SAVE]
     - 🧵 ➔ [MERGE]
     - ➔ ➔ ->
   - Visualizer Icons:
     - 🧮 ➔ [CALC]
     - 📈 ➔ [PLOT]
     - 📂 ➔ [FILE]
     - 📅 ➔ [DATE]

---  

poetry run code2prompt --path "C:\Users\ping\Files_win10\python\py311\stocks\notebooks_RLVR_v2\tests\" --exclude "**/__pycache__/**,**/__init__.py" | Out-File -Encoding utf8 tests_dir.txt


poetry run code2prompt --path "C:\Users\ping\Files_win10\python\py311\stocks\notebooks_RLVR_v2\verify_UI_n_features_calc_v3.ipynb" --exclude "**/__pycache__/**,**/__init__.py" | Out-File -Encoding utf8 verify_calc.txt

C:\Users\ping\Files_win10\python\py311\stocks\notebooks_RLVR_v2\verify_UI_n_features_calc_v3.ipynb


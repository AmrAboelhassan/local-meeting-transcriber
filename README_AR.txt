Local Meeting Transcriber - Version 5
====================================

الفكرة:
التطبيق يشتغل على جهاز ويندوز محليًا بواجهة بسيطة.
الشخص يرفع تسجيل اجتماع أو محاضرة أو مقابلة، والتطبيق يطلع transcript.txt و subtitles.srt.

الترتيب الصحيح أول مرة:

1) فك الضغط عن الملف ZIP.
2) افتح فولدر التطبيق.
3) شغل:
   00_INSTALL_APP_FIRST.bat

ده هيعمل:
- يثبت Python 3.11 لو مش موجود.
- يعمل .venv للتطبيق.
- يثبت مكتبات التطبيق.
- يثبت مكتبات NVIDIA cuDNN/cuBLAS داخل بيئة التطبيق.

4) لو عايز GPU وظهر Error فيه CUDA/cublas/cudnn:
   شغل مرة واحدة:
   01_INSTALL_CUDA_12_FOR_GPU.bat

ده هينزل ويفتح installer الرسمي:
   cuda_12.0.0_527.41_windows.exe

بعد CUDA يفضل تعمل Restart للجهاز.

5) لتشغيل التطبيق كل مرة:
   02_START_APP.bat

داخل التطبيق:

1) اختار الموديل:
   Balanced - medium recommended default

2) دوس:
   Download selected model

دي خطوة منفصلة ومهمة.
الموديل بيتحمل مرة واحدة فقط.
بعد كده مش هيحمله تاني.

3) بعد ما يكتب إن الموديل Ready:
   ارفع تسجيل الاجتماع أو المحاضرة أو المقابلة.

4) اختار:
   Language: Arabic أو Auto detect
   Device: Auto: try GPU then CPU
   Compute: Auto
   Chunk length: 20 minutes
   Clean/normalize audio: ON
   VAD filter: ON

5) دوس:
   Start Transcription

النتائج هتظهر للتحميل:
- transcript.txt
- clean_transcript.txt
- subtitles.srt
- process_log.txt

ملاحظات مهمة:

- لو أول مرة الترانسكريبت كان واقف على Downloading، ده طبيعي زمان. في النسخة دي التحميل اتفصل في زر لوحده.
- لو GPU عمل مشكلة، التطبيق ممكن يرجع CPU تلقائيًا في Auto mode.
- CPU أبطأ لكنه يشتغل بدون CUDA.
- للموديل large-v3 ممكن يحصل Memory Error على كروت الشاشة ذات الذاكرة المحدودة. لو حصل استخدم medium.
- خلي مساحة فاضية 10-15GB على الأقل.

إعدادات مقترحة كبداية لمعظم الأجهزة:
- Model: Balanced - medium
- Device: Auto
- Compute: Auto
- Chunk: 20 minutes

لو عايز جودة أعلى جرب large-v3، لكن medium هو الاختيار العملي.

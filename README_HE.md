# Hebrew OCR Unified Builder v11

כלי fail-closed לבניית מאגר OCR/HTR עברי מאוחד מארבעת המאגרים הפרטיים בחשבון `ssdataanalysis`:

- `hebrew-ocr-foundation-v1`
- `hebrew-htr-curated-v1`
- `hebrew-ocr-corpus`
- `hebrew-architecture-corpus`

ה־builder אינו מבצע concat עיוור. הוא מנרמל Unicode, שומר תוויות ב־logical order, נועל revisions ופונטים, מסיר כפילויות, מונע leakage בין splits, יוצר דוגמאות חדשות מטקסט Tier-A של קורפוס האדריכלות, ומפריד פיזית בין שכבות אמון.

## שכבות הנתונים

- **gold** — ברירת המחדל לאימון; תוויות אנושיות מאומתות או ground truth סינתטי בעל provenance ברור.
- **extended** — opt-in; חומר שימושי אך פחות ודאי, כגון diffusion או Tier-B.
- **quarantine** — ביקורת בלבד; משקל אימון אפס ואינו נכלל ב־config הראשי.

מילים, תווים בודדים ועמודים מלאים נשמרים ב־configs נפרדים כדי שלא יתערבבו בטעות עם שורות recognition.

## הפעלה

```bash
chmod +x RUN_ME.command
./RUN_ME.command
```

הפקודה מבצעת:

1. אימות חשבון Hugging Face.
2. יצירת venv והתקנת dependencies נעולות.
3. הרצת כל בדיקות הקוד, כולל PyArrow ו־corruption suite.
4. mini-build אמיתי מכל ארבעת המקורות וכל splits/configs.
5. QA מקיף ויצירת `LOCAL_READY.json` ל־mini בלבד אם הכול עבר.
6. build מלא resumable.
7. QA, previews, release manifest וניסויי השחתה על הפלט המלא.
8. העלאה ל־dataset פרטי ואימות הורדה חוזרת.
9. יצירת `REMOTE_READY.json` רק לאחר התאמה מלאה של המאגר המרוחק.

קובצי העבודה נשמרים ב־`~/hebrew-ocr-unified-work-v11`. אפשר להריץ שוב לאחר ניתוק; completed source units נבדקים מחדש לפי hash ומספר שורות לפני resume. זהות הבנייה קשורה לא רק ל־config ול־revisions אלא גם ל־SHA-256 דטרמיניסטי של כל קוד ה־builder ושל קובצי התלויות הנעולים. שינוי קוד עם אותו מספר גרסה אינו יכול להמשיך build ישן בשקט.

## חותמות מוכנות

- `LOCAL_READY.json` — נוצר רק אחרי QA מקומי, previews, corruption suite ו־release manifest.
- `REMOTE_READY.json` — נוצר רק אחרי אימות המאגר הפרטי המרוחק.
- `qa/QA_REPORT.json`
- `qa/CORRUPTION_REPORT.json`
- `RELEASE_MANIFEST.json`
- `CHECKSUMS.sha256`
- `previews/PREVIEW_INVENTORY.json`
- `VERIFIED_POINTED_AUDIT.json` — קושר את קורפוס הניקוד למניפסט, ל־revision ול־SHA-256 המדויקים שלו.

מסלול `verified_pointed_rerender` מפיק מחדש יותר מ־100 אלף שורות מנוקדות מ־ground truth לוגי מאומת, בעוד שתמונות המקור Tier-B נשארות ב־extended ואינן מקודמות ל־gold.

חומר Architecture שמקורו בסריקות OCR קודמות אינו משמש כתווית זהב. כל מקטע Tier-A נקי חייב להיכנס כפלט קנוני או לקבל outcome מפורש ב־ledger; כשל רינדור של Tier-A מפיל את הבנייה במקום להיעלם בשקט.

## גבול הטענה

המאגר נבנה להיות SOTA-capable ומקור אימון יחיד, אך טענת SOTA מחייבת אימון מודל והשוואה על benchmark אנושי נעול. ה־builder אינו מזייף את ההבחנה הזאת.

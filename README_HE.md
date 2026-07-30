# Hebrew OCR Unified Builder v13

כלי fail-closed לבניית מאגר OCR/HTR עברי מאוחד מארבעת המאגרים הפרטיים בחשבון `ssdataanalysis`:

- `hebrew-ocr-foundation-v1`
- `hebrew-htr-curated-v1`
- `hebrew-ocr-corpus`
- `hebrew-architecture-corpus`

ה־builder אינו מבצע concat עיוור. הוא מנרמל Unicode, שומר תוויות ב־logical order, נועל revisions ופונטים, מסיר כפילויות, מונע leakage בין splits, יוצר דוגמאות חדשות מטקסט Tier-A של קורפוס האדריכלות, ומפריד פיזית בין שכבות אמון.

## שכבות הנתונים

- **gold** — ברירת המחדל לאימון; תוויות אנושיות מאומתות או ground truth סינתטי בעל provenance ברור.
- **extended** — opt-in; חומר שימושי אך פחות ודאי, כגון diffusion, Tier-B וכתב שומרוני שתוויותיו Unicode עברי אך צורותיו החזותיות אינן כתב עברי מרובע רגיל.
- **quarantine** — ביקורת בלבד; משקל אימון אפס ואינו נכלל ב־config הראשי.

מילים, תווים בודדים ועמודים מלאים נשמרים ב־configs נפרדים כדי שלא יתערבבו בטעות עם שורות recognition. לצד views מאוחדים, כל config פיזי נחשף בנפרד לצורך curriculum, benchmarking ודגימה מבוקרת.

## דרישות

- macOS או Linux.
- Python **3.12.x בלבד**.
- Hugging Face CLI מחובר לחשבון `ssdataanalysis` עם הרשאת write.
- לפחות **200 GiB** פנויים בתיקיית העבודה.

ב־macOS:

```bash
brew install python@3.12
hf auth login
```

## הפעלה

```bash
chmod +x RUN_ME.command
./RUN_ME.command
```

לבדיקת הרשאת הכתיבה בלבד, בלי להתחיל build:

```bash
.venv/bin/python -m heocr_unified probe-upload --config config.json
```

הפקודה מבצעת:

1. אימות חשבון Hugging Face וסביבת Python 3.12.
2. יצירת venv והתקנת dependencies נעולות.
3. הרצת כל בדיקות הקוד, כולל PyArrow ו־corruption suite.
4. בדיקת הרשאת write אמיתית: יצירת repo פרטי זמני, העלאת קובץ, הורדה חוזרת ומחיקה.
5. mini-build אמיתי מכל ארבעת המקורות וכל splits/configs.
6. QA מקיף ויצירת `LOCAL_READY.json` ל־mini רק אם הכול עבר.
7. build מלא resumable.
8. QA, previews, release manifest וניסויי השחתה על הפלט המלא.
9. העלאה ל־dataset פרטי ואימות הורדה חוזרת.
10. יצירת `REMOTE_READY.json` רק לאחר התאמה מלאה של המאגר המרוחק.

קובצי העבודה נשמרים ב־`~/hebrew-ocr-unified-work-v13`. אפשר להריץ שוב לאחר ניתוק; completed source units נבדקים מחדש לפי hash ומספר שורות לפני resume. זהות הבנייה קשורה ל־config, ל־revisions, לפונטים ול־SHA-256 דטרמיניסטי של קוד ה־builder וקובצי התלויות.

## כיסוי סינתטי חדש

- כל טקסט Architecture Tier-A נקי מקבל דוגמת primary קנונית או outcome מפורש ב־ledger.
- כ־22% משורות Architecture train מקבלות וריאציית רינדור נוספת דטרמיניסטית; validation/test נשארים עם מופע קנוני יחיד.
- נוצרות **300,000** שורות מקצועיות מובנות עם מספרים, יחידות, קנ״מ, גוש/חלקה, מפלסים, קואורדינטות ו־mixed BiDi.
- נוצרות לפחות 100,000 שורות מנוקדות חדשות במסלול `verified_pointed_rerender`, מהתמלול הלוגי המאומת בלבד. תמונות Tier-B המקוריות אינן מקודמות ל־gold.
- עמודים מלאים נשמרים עם bounding boxes, polygons, baselines ו־reading order.
- חלוקת משפחות הפונט בין train/validation/test נשארת מבודדת, ובכל split נשמר לפחות פונט נעול אחד עם כיסוי מלא של ניקוד, Meteg ו־Sof Pasuq.

## חותמות מוכנות

- `LOCAL_READY.json` — נוצר רק אחרי QA מקומי, previews, corruption suite ו־release manifest.
- `REMOTE_READY.json` — נוצר רק אחרי אימות המאגר הפרטי המרוחק.
- `qa/QA_REPORT.json`
- `qa/CORRUPTION_REPORT.json`
- `RELEASE_MANIFEST.json`
- `CHECKSUMS.sha256`
- `previews/PREVIEW_INVENTORY.json`
- `VERIFIED_POINTED_AUDIT.json`
- `ARCHITECTURE_TEXT_RESOLVER.json`
- `EVALUATION_RESERVATIONS.json`

## גבול הטענה

המאגר נבנה להיות SOTA-capable ומקור אימון יחיד, אך טענת SOTA מחייבת אימון מודל והשוואה על benchmark אנושי נעול. ה־builder אינו מזייף את ההבחנה הזאת.

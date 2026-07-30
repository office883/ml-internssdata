# Hebrew OCR Unified Builder v15

כלי fail-closed לבניית מאגר OCR/HTR עברי מאוחד מארבעת המאגרים הפרטיים בחשבון `ssdataanalysis`:

- `hebrew-ocr-foundation-v1`
- `hebrew-htr-curated-v1`
- `hebrew-ocr-corpus`
- `hebrew-architecture-corpus`

ה־builder מנרמל Unicode ושומר logical RTL, מסיר כפילויות, מונע leakage בין splits, יוצר תמונות חדשות מטקסט Architecture Tier-A, ומפריד פיזית בין `gold`,‏ `extended` ו־`quarantine`.

## מה תוקן ב־v15

- פונט הגיבוי של macOS בשם `LastResort` נפסל במפורש ואינו יכול לשמש לרינדור, smoke tests או בדיקות.
- בדיקת macOS משתמשת באותו מנגנון גילוי פונטים מסונן שבו משתמש ה־builder עצמו.
- עמודים מלאים מתאימים את גודל הפונט באופן דטרמיניסטי כאשר עטיפה רגילה אינה נכנסת לקנבס.
- גובה שורות בטבלאות ובטפסים מחושב לפי הגאומטריה האמיתית של כל fragment, לא לפי קבוע משוער.
- כל fragment שומר `source_line_index`,‏ `fragment_index` ו־`fragment_count` לצד reading order רציף.
- נוספו בדיקות רגרסיה לעמודי Architecture בגודל המרבי המותר ול־LastResort ב־macOS.

## דרישות

- macOS או Linux.
- Python **3.12.x בלבד**.
- Hugging Face CLI מחובר לחשבון `ssdataanalysis` עם הרשאת write.
- לפחות **200 GiB** פנויים.

## הפעלה

```bash
chmod +x RUN_ME.command && ./RUN_ME.command
```

הסקריפט מתקין ומוודא RAQM, HarfBuzz ו־FriBiDi; בונה Pillow מהמקור ב־macOS; מריץ את כל הבדיקות ללא skip; בודק הרשאת כתיבה פרטית; מבצע mini-build אמיתי; ורק לאחר PASS בונה ומעלה את המאגר המלא.

קובצי העבודה נשמרים ב־:

```text
~/hebrew-ocr-unified-work-v15
```

הרצה חוזרת ממשיכה ממצב מאומת באמצעות fingerprints, hashes, source revisions ומספרי שורות.

## פלט

היעד הוא Dataset פרטי אחד:

```text
ssdataanalysis/hebrew-ocr-unified-sota-v1
```

הבנייה מסומנת מוכנה רק לאחר יצירת `LOCAL_READY.json`; המאגר המרוחק מסומן מוכן רק לאחר יצירת `REMOTE_READY.json` ואימות הורדה חוזרת.

## גבול הטענה

המאגר נבנה להיות SOTA-capable ומקור אימון יחיד. טענת SOTA בפועל עדיין מחייבת אימון מודל והשוואת CER/Grapheme-CER על benchmark אנושי נעול.

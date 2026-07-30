# macOS runtime — Hebrew OCR Unified Builder v15

ה־launcher מתקין באמצעות Homebrew את Python 3.12,‏ libraqm,‏ HarfBuzz,‏ FriBiDi,‏ FreeType וספריות תמונה. לאחר מכן Pillow 12.2.0 נבנה מהמקור בתוך `.venv`, כדי להבטיח ש־RAQM פעיל בפועל.

הבדיקה המקדימה משתמשת ב־`heocr_unified.fonts.discover_fonts`, ולכן פונטים גנריים מסוג `LastResort`,‏ `Apple Symbols` ו־`Apple Color Emoji` אינם מתקבלים כפונטים עבריים.

ההרצה אינה ממשיכה אם:

- RAQM אינו פעיל;
- אין פונט אמיתי שתומך בעברית, ניקוד, ספרות ו־Latin;
- רינדור RTL/mixed-BiDi או WebP round-trip נכשל;
- אחת מבדיקות הקוד נכשלת או נרשמת כ־skipped.

הפעלה:

```bash
chmod +x RUN_ME.command && ./RUN_ME.command
```

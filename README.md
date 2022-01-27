# Crime Named Entity Recognition

This project aims to label entities in Thai crime-related texts using data scraped from Thai news websites e.g. www.thairath.co.th www.khaosod.co.th

## Problem statement
- To extract insight entity from text and help officer know what the components are in the ridiculously long texts
- To use an NER model for Relation extraction task for further development

## Label
1. PERSON : A name of person (including prefix) e.g. นายสมชาย พ.ต.อ.สมชาย แซ่ตั้ง น้องสมชาย
2. ORGANIZATION : An organization or a group of the purposed e.g. มูลนิธีกู้ภัยสว่างประทีป บริษัท เอ จำกัด โรงพยาบาลพระราม 9
3. LOCATION : A spontaneous place e.g. ถนนสุขุมวิท แม่น้ำเจ้าพระยา บ้านเลขที่ 1 ต.มาบข่า อ.เมือง จ.ระยอง แยกห้วยขวาง
4. TIME : Time
5. DATE : Date
6. VEHICLE : Vehicle name
7. OBJECT : An object which is found in crime scene e.g.ปลอกกระสุน ถุงมือ ยาไอซ์ 
8. WEAPON : A weapon e.g. ปืนสั้น ปืนขนาด .38
9. COLOR : A color e.g. สีขาว สีดำ 
10. LP : A license plate e.g. กจ 1234 ระยอง

# 🌾 FikrimGelecek VerimGören

**Geliştiriciler:**  
👩‍💻 **İrem Morkoç**  
👨‍💻 **Mehmet Yalaz**

---

## 📘 Proje Tanımı

**VerimGören**, FikrimGelecek programı kapsamında geliştirilen bir **tarımsal veri analizi ve karar destek sistemidir.**  
Proje; doğru **ürün – yer – zaman** eşleşmesiyle tarımsal üretimi daha verimli, çevre dostu ve sürdürülebilir hale getirmeyi hedefler.

VerimGören, çiftçilere ve tarımsal paydaşlara;  
iklim, toprak, rakım, ışık yoğunluğu gibi çevresel verileri analiz ederek **en uygun ürün seçimi** ve **verim tahmini** yapma imkânı sunar.

---

## ⚠️ Veri Notu

Projede kullanılan veri setleri (iklim rasterleri, toprak veritabanı, gece ışığı, SRTM rakım verileri vb.) oldukça büyük boyutlardadır.  
Bu nedenle, **GitHub veri yükleme sınırlarını aşmamak için tam veriler bu depoda yer almamaktadır.**  
Burada yalnızca proje yapısını, örnek notebook’ları, CSV formatında bitki parametre tablolarını ve Streamlit tabanlı arayüz kodlarını görebilirsiniz.

---

## 🌱 Projenin Amacı

- Tarımsal üretimde **verimliliği artırmak**,  
- **Su ve gübre kullanımını optimize etmek**,  
- Kaynak israfını önleyerek **karbon salımını azaltmak**,  
- Kırsal bölgelerde yaşayan üreticilerin dijital dönüşümüne katkı sağlamak.

---

## 🧩 Temel Bileşenler

| Bileşen | Açıklama |
|---------|-----------|
| `app.py` | Streamlit tabanlı analiz arayüzü |
| `notebooks/` | Uygulama prototipleri ve veri analizi notebook’ları |
| `hwsd_data/` | Toprak verileri (HWSD) |
| `data/climate/` | Uydu tabanlı iklim verileri |
| `VerimGoren_Bitki_Parametreleri_Tam.csv` | Bitki uygunluk parametreleri tablosu |

---

## 🔬 Çalışma Mantığı

1. Kullanıcı konum bilgisi girer.  
2. Sistem bu konuma ait **iklim, toprak, rakım ve ışık verilerini** analiz eder.  
3. Analiz sonuçlarına göre **uygun ürün listesi** ve **verim tahmini** oluşturulur.  
4. Sonuçlar kullanıcıya görsel olarak sade bir arayüzde sunulur.

---

## 🌍 Sürdürülebilirlik Katkısı

- 💧 %30’a kadar **su tasarrufu**  
- 🌿 Gereksiz **gübre ve enerji kullanımının azaltılması**  
- 🌾 **Toprak sağlığının korunması**  
- 🌎 **Karbon ayak izinin düşürülmesi**

---

## 🧠 Teknolojiler ve Araçlar

- **Python** (NumPy, Pandas, Rasterio, Streamlit, Matplotlib)
- **Uydu Verileri:** NASA POWER, SRTM, VIIRS
- **Veri Kaynağı:** HWSD (FAO)
- **Arayüz:** Streamlit
- **Depolama:** CSV, GeoTIFF, MDB (Microsoft Access)

---

## 👩‍🔬 Geliştiriciler Hakkında

- **İrem Morkoç:** Tarım teknolojileri, dijital dönüşüm ve veri odaklı üretim üzerine çalışan bir geliştirici.  
- **Mehmet Yalaz:** Tarımsal analiz, yazılım geliştirme ve coğrafi veri entegrasyonu alanında uzmanlaşmış veri bilimci.

---


---

> Bu proje, FikrimGelecek girişimi kapsamında “dijital tarımda yerli inovasyon” hedefiyle geliştirilmiştir.  
> VerimGören — **Verim artar, israf azalır.**

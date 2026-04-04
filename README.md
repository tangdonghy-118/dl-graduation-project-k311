
# Real Estate Hybrid Recommender & Clustering System
> **Đồ án Tốt nghiệp - K311 | Data Science & Machine Learning**

---

```markdown
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```
Hệ thống tích hợp hỗ trợ người dùng tìm kiếm, nhận đề xuất và định vị phân khúc bất động sản thông qua các thuật toán Học máy tối ưu. Ứng dụng giải quyết bài toán đa tiêu chí (Multi-criteria) giúp khách hàng tìm được căn nhà mơ ước dựa trên sự cân bằng giữa **Vị trí - Giá cả - Đặc điểm**.

---

## Tính năng cốt lõi

### 1. Hệ thống Gợi ý Kết Hợp (Hybrid Recommender)
Động cơ gợi ý thông minh kết hợp 3 thành phần chính để tính toán điểm tương đồng (`hybrid_score`):
* **NLP Content-based:** Phân tích ngữ nghĩa mô tả nhà bằng **TF-IDF Vectorization** và **Cosine Similarity**.
* **Price Similarity:** Sử dụng hàm suy giảm hàm mũ (Exponential Decay) để đo lường độ khớp ngân sách.
* **Location Similarity:** So sánh vị trí thông qua kỹ thuật **One-hot Encoding** trên các Quận/Huyện.
* **Dynamic Weighting:** Cơ chế cho phép người dùng điều chỉnh trọng số ưu tiên ngay tại thời gian thực.

### 2. Phân cụm Thị trường (K-Means Clustering)
* **Segmentation:** Nhóm bất động sản thành 3 phân khúc chủ đạo: **Phổ thông, Trung cấp, Cao cấp**.
* **Asset Positioning:** Dự báo phân khúc cho tài sản cá nhân và so sánh với giá trị trung bình của thị trường.
* **Interactive Analytics:** Hệ thống hóa dữ liệu qua biểu đồ Elbow, Silhouette và Scatter Map động.

---

## Kiến trúc Hệ thống & Luồng Dữ liệu

Dự án được thiết kế theo kiến trúc module hóa để đảm bảo tính mở rộng:

```text
nhatot_recommender_and_clustering/
├── app.py                              # Giao diện chính
├── requirements.txt                    # Thư viện cần thiết
├── setup.sh                            # Script cài đặt môi trường
├── Procfile                            # Cấu hình triển khai
├── .vscode/                            # Cấu hình VSCode
│   └── settings.json                   
├── utils/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── ham_data_preprocessing.py       # Các hàm xử lý cho clustering
│   ├── recommender.py                  # Các hàm tiền xử lý và hàm tạo gợi ý (dùng cho recommender)
│   └── ui.py                           # Định dạng UI (CSS, Cards)
├── models/
│   ├── nha_cosine_sim.pkl              # Ma trận cosine similarity
│   └── kmeans.pkl                      # Model KMeans cho clustering
├── notebooks/
│   ├── DL07_P2_Recommender.ipynb       # Train model for recommender
│   ├── DL08_P2_Clustering.ipynb        # Train model for clustering
│   ├── Data_preprocessing_DL07.ipynb   # Tiền xử lý dữ liệu cho clustering
│   └── notebook2/                      # Hỗ trợ cho clustering
│       ├── clustering_utils.py         
│       └── visualization_utils.py
├── images/                             # Chứa các hình ảnh dùng trong UI
│   ├── nhatot.png
│   ├── nhatot.jpg
│   └── banner_nhatot.png
└── Data/
    ├── cleaned_data_recommend.csv      # Dữ liệu đã được tiền xử lý cho recommender
    ├── quan-binh-thanh.csv             # Dữ liệu thô khu vực quận Bình Thạnh
    ├── quan-go-vap.csv                 # Dữ liệu thô khu vực quận Gò Vấp
    ├── quan-phu-nhuan.csv              # Dữ liệu thô khu vực quận Phú Nhuận
    ├── well_formed_data.csv            # Cho clustering
    └── files/                          # Chứa các file txt dùng cho NLP 
        ├── emojicon.txt    
        ├── english-vnmese.txt
        ├── teencode.txt
        ├── vietnamese-stopwords.txt
        └── wrong-word.txt
```

### 🛠 Công nghệ sử dụng
| Lĩnh vực | Công nghệ |
| :--- | :--- |
| **User Interface** | Streamlit, HTML5/CSS3 (Custom Theme) |
| **Data Preprocessing** | Pandas, Numpy |
| **ML Models** | Scikit-learn (K-Means, TF-IDF, Cosine Similarity) |
| **Visualization** | Plotly Express, Seaborn, Matplotlib |

---

## Tối ưu hóa hiệu năng (Performance Optimization)

Nhằm đảm bảo ứng dụng vận hành mượt mà trên các môi trường giới hạn tài nguyên (như Streamlit Cloud):
* **RAM Management:** Áp dụng chiến lược xóa biến tạm (`del`) và cưỡng bức giải phóng bộ nhớ (`gc.collect()`) sau các tác vụ nặng (như render Plotly hoặc xử lý ma trận).
* **Smart Caching:** Sử dụng `@st.cache_data` để nạp các tập dữ liệu lớn và mô hình vào bộ nhớ đệm, giúp giảm thời gian phản hồi từ giây xuống mili-giây.
* **Data Sampling:** Tối ưu hóa render đồ họa bằng cách lấy mẫu dữ liệu thông minh (Sampling) khi trực quan hóa hàng chục nghìn điểm dữ liệu.

---

## Cài đặt & Sử dụng

### 1. Clone Repository
```bash
git clone [https://github.com/tangdonghy-118/dl-graduation-project-k311.git](https://github.com/tangdonghy-118/dl-graduation-project-k311.git)
cd dl-graduation-project-k311
```

### 2. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy App
```bash
streamlit run app.py
```

---

## 👥 Đội ngũ phát triển

* **Phan Đặng Anh** - *Project Leader & Architect*
* **Tang Đông Hy** - *Recommender System Specialist*
* **Phó Quốc Dũng** - *Clustering & Data Analysis*
* *Giáo viên hướng dẫn -  **Khuất Thùy Phương***

---
© 2026 Graduation Project - K311. Bảo lưu mọi quyền.
```
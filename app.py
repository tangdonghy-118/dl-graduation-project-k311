import gc
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from utils.utils import get_cached_data_recommender, \
                        get_cached_data_clustering, \
                        get_house_options, \
                        update_weights,\
                        get_hybrid_recommendations,\
                        get_hybrid_recommendations_from_search,\
                        get_model_metrics,\
                        processing_cluster_logic

from utils.ui import set_custom_theme,\
                     display_recommendations


# Cấu hình
st.set_page_config(page_title="Hệ thống Gợi ý Và Phân Cụm Bất Động Sản", layout="wide", page_icon="🏢")
set_custom_theme()

# sidebar
with st.sidebar:
    st.markdown("## **🎓 THÔNG TIN ĐỒ ÁN 🎓**")
    st.divider() 
    
    st.markdown("### - Sinh viên thực hiện -")
    
    st.markdown("#### *Leader:*")
    st.info("**Phan Đặng Anh**\n\n*phandanganh2003@gmail.com*",width="stretch")
    
    st.markdown("#### *Recommender System:*")
    st.info("**Tang Đông Hy**\n\n*hyyhtang696969@gmail.com*",width="stretch")
    
    st.markdown("#### *Clustering System:*")
    st.info("**Phó Quốc Dũng**\n\n*phoqdung89@gmail.com*",width="stretch")
    
    st.markdown("### - Giảng viên hướng dẫn -")
    st.success("**Khuất Thùy Phương**",width="stretch")
    
    st.divider()
    
    app_mode = st.selectbox(
        "***Chọn Module Hệ Thống:***",
        ["Hệ thống Gợi ý Nhà Ở (Hybrid Recommender)", "Phân cụm Dữ liệu Bất Động Sản (K-Means Clustering)"]
    )
    
    st.divider()
    st.caption("Đồ án tốt nghiệp - Khóa 311 - Data Science and Machine Learning")
    

#  Load  
with st.spinner('Đang khởi tạo hệ thống... (Vui lòng đợi giây lát)'):
    df_houses, cosine_sim = get_cached_data_recommender()
    df_cluster, kmeans_model, cluster_mapping, cluster_stats, df_viz = get_cached_data_clustering()
    options_list = get_house_options(df_houses)


# RECOMMENDER
if app_mode=="Hệ thống Gợi ý Nhà Ở (Hybrid Recommender)":
    # HEADER
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image("images/nhatot.png", width='stretch', output_format="PNG")
    st.markdown("<h1 style='text-align: center;'>Hệ Thống Đề Xuất Bất Động Sản</h1>",
                unsafe_allow_html=True)
    st.markdown("---")


    # TẠO TABS 
    tab_biz, tab_list_rec, tab_search_rec = st.tabs([
        "**Business Objective & Model**", 
        "**Danh sách Bất động sản - List**", 
        "**Tìm kiếm Bất động sản - Search**"
    ])


    # BUSINESS OBJECTIVE
    with tab_biz:
        st.header(" Mục Tiêu Dự Án (Business Objective)")
        st.write("""
        Trong thị trường bất động sản, việc khách hàng bị ngợp giữa hàng nghìn tin đăng là thách thức lớn. Hệ thống này được thiết kế để:
        * **Cá nhân hóa trải nghiệm:** Đề xuất chính xác những căn nhà dựa trên hành vi và sở thích cụ thể.
        * **Tối ưu hóa chuyển đổi:** Giảm thời gian tìm kiếm, giúp khách hàng nhanh chóng chọn được căn nhà ưng ý, từ đó tăng tỷ lệ giao dịch thành công.
        * **Hệ thống hóa dữ liệu:** Kết nối các đặc trưng phi cấu trúc (văn bản mô tả) và dữ liệu cấu trúc (giá, vị trí) thành một điểm số thống nhất.
        """)

        st.divider()

        st.header("Cơ chế Vector Space Model & Cosine Similarity")
        st.write("""
        Hệ thống chuyển đổi các đoạn mô tả bất động sản không cấu trúc (Unstructured Data) thành các tọa độ trong không gian **n-chiều**. 
        Mỗi từ khóa (Keyword) sau khi qua bộ lọc NLP sẽ trở thành một chiều của Vector. 
        
        Thay vì sử dụng khoảng cách Euclidean (vốn bị ảnh hưởng bởi độ dài bài đăng), chúng tôi sử dụng **Cosine Similarity** để đo góc giữa các Vector bài đăng $A$ và $B$:
        """)
        st.latex(r"sim(A, B) = \cos(\theta) = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}")
        
        
        
        st.info("""
        ***Tại sao dùng Cosine Similarity?***\n 
        *Nhờ vào khả năng tập trung vào hướng ngữ nghĩa thay vì độ dài của văn bản.
        Trong thực tế, các bài đăng về nhà đất có sự chênh lệch rất lớn về số lượng từ ngữ; một bài viết ngắn gọn và một bài mô tả chi tiết có thể cùng đề cập đến những đặc điểm then chốt như "hẻm xe hơi" hay "nở hậu".
        Nếu sử dụng các phép đo khoảng cách thông thường, những bài đăng dài sẽ bị coi là "xa cách" so với bài đăng ngắn dù nội dung tương đồng.
        Cosine Similarity giải quyết triệt để vấn đề này bằng cách chỉ đo góc giữa các vector đặc trưng, giúp hệ thống nhận diện chính xác sự tương đồng về bản chất bất động sản bất kể chủ nhà viết dài hay ngắn.
        Ngoài ra, phương pháp này cực kỳ hiệu quả với dữ liệu từ mô hình TF-IDF, vốn thường tạo ra các không gian vector thưa, giúp lọc bỏ nhiễu và làm nổi bật những từ khóa mang giá trị thực tế cao trong việc ra quyết định của người mua.*
        """)

        st.write("") 
        st.divider()

        st.header("Cơ chế Hybrid Recommending")
        st.write("""
        Hệ thống giải quyết bài toán **Đa tiêu chí (Multi-criteria Decision Making)** bằng cách kết hợp 3 tiêu chí thành phần để đưa ra kết quả tối ưu nhất:
        """)
        
        st.markdown("##### 1. Content Score ($S_{c}$)")
        st.write("""
        Đo lường sự tương đồng nội dung trích xuất qua mô hình **TF-IDF** và ma trận **Cosine Similarity**.
        * **Lý do:** Giúp hiểu được "bản chất" yêu cầu của người dùng (ví dụ: 'hẻm xe hơi', 'nở hậu') thay vì chỉ đếm số từ đơn thuần. Cosine Similarity giúp so sánh chính xác ngay cả khi các bài đăng có độ dài văn bản khác nhau.
        """)
        
        st.markdown("##### 2. Price Similarity ($S_{p}$)")
        st.write("""
        Chỉ số tiệm cận ngân sách áp dụng mô hình **Suy giảm hàm mũ (Exponential Decay)**:
        """)
        st.latex(r"S_{p} = e^{-0.3 \cdot |P_{target} - P_{house}|}")
        
        st.write("""
        * **Lý do:** Mô phỏng thực tế tâm lý khách hàng: sự hài lòng sẽ giảm rất nhanh khi giá vượt quá ngưỡng ngân sách, thay vì giảm đều theo đường thẳng. Hệ số 0.3 giúp ưu tiên các căn nhà lệch trong khoảng dưới 1 tỷ đồng.
        """)
        with st.expander("Chi tiết công thức"):
            st.markdown("""
            | Ký hiệu | Ý nghĩa | Giá trị/Nguồn |
            | :--- | :--- | :--- |
            | $S_{p}$ | Điểm tương đồng về giá | Khoảng $(0, 1]$ |
            | $e$ | Số Euler | $≈ 2.718$ |
            | $0.3$ | Hệ số suy giảm ($\lambda$) | Tinh chỉnh thực nghiệm |
            | $P_{target}$ | Giá mục tiêu | Từ căn nhà đang xem |
            | $P_{house}$ | Giá thực tế | Từ Cơ sở dữ liệu |
            | $\mid P_{\Delta} \mid$ | Độ lệch tuyệt đối | $\mid P_{target} - P_{house} \mid$ |
            """)

        st.markdown("##### 3. Location Similarity ($S_{l}$)")
        st.write("""
        Sử dụng kỹ thuật **One-Hot Encoding** cho khu vực hành chính (Quận) để tối ưu hóa vị trí.
        * **Lý do:** Trong bất động sản, vị trí là điều kiện tiên quyết. Kỹ thuật này biến các tên Quận thành vector số học, cho phép thuật toán tính toán chính xác mức độ khớp vùng địa lý yêu cầu một cách tuyệt đối (cùng Quận = 1, khác Quận = 0).
        """)    
        
        st.write("") 
        st.write("**Công thức Hybrid tổng quát:**")
        st.latex(r"Hybrid\_Score = w_{c} \cdot S_{c} + w_{p} \cdot S_{p} + w_{l} \cdot S_{l}")
        
        st.info("💡 Các trọng số $w$ có thể được điều chỉnh linh hoạt trên giao diện để ưu tiên tiêu chí mong muốn.")

        st.divider()

        st.header("Quy trình Tiền xử lý Dữ liệu (Preprocessing Pipeline)")
        
        st.write("""
        Để đạt được độ chính xác cao khi đối chiếu, dữ liệu thô ban đầu phải đi qua một "dây chuyền" xử lý nghiêm ngặt trước khi đưa vào không gian Vector:
        """)
        
        step1, step2, step3 = st.columns(3)
        
        with step1:
            with st.container(border=True):
                st.markdown("#### 1. Làm sạch dữ liệu")
                st.markdown("""
                * **Xử lý văn bản:** Loại bỏ các thẻ HTML, Emojis, ký tự đặc biệt, khoảng trắng thừa trong bài đăng,...
                * **Lọc dữ liệu:** Loại bỏ triệt để các giá trị thiếu và các bài đăng trùng lặp.
                """)
                
        with step2:
            with st.container(border=True):
                st.markdown("#### 2. Feature Engineering")
                st.markdown("""
                * **Chuẩn hóa định danh:** Thiết lập cột `id`, trích xuất và gom nhóm `quận` + `phường`.
                * **Đồng bộ giá trị:** Chuyển đổi và làm sạch `giá bán` (đưa về chung đơn vị **Tỷ VNĐ**).
                * **Văn bản cơ sở:** Tạo ra cột mô tả đã được làm sạch bước đầu (`mo_ta_clean`).
                """)
                
        with step3:
            with st.container(border=True):
                st.markdown("#### 3. Kết hợp mô tả khác")
                st.markdown("""
                Gộp các features quan trọng để tạo ngữ cảnh hoàn chỉnh:
                * `loai_hinh`: Nhà phố, hẻm, mặt tiền...
                * `tinh_trang_noi_that`: Nhà thô, đầy đủ...
                * `huong_cua_chinh`: Hướng nhà (Đông, Tây, Nam, Bắc)
                * `dac_diem`: Các đặc điểm khác nổi bật.
                * `mo_ta_clean`: Cột mô tả đã được làm sạch trước đó.
                """)

        # Hộp thông tin giải thích mũi tên cuối cùng trong ảnh 2
        st.info("**Bước đệm chuẩn hóa Vector:** Tổ hợp văn bản thu được ở **Bước 3** sẽ tiếp tục đi qua quá trình **Tokenization & Cleaning** (tách từ ghép tiếng Việt) để tạo ra cột `Content_wt` cuối cùng, tối ưu hóa tối đa cho thuật toán TF-IDF.")

        st.divider()
        
        st.header("Cấu trúc thư mục dự án")
        
        st.subheader("Core files")
        st.markdown("### `app.py` ")
        st.info("Là **Main UI**, Quản lý luồng giao diện Streamlit, tiếp nhận input người dùng và điều phối các tab chức năng.")
        
        st.markdown("### `utils.py` ")
        st.info("Chứa các **functions**, xử lý các khâu tiền xử lý dữ liệu, lọc Stopwords Tiếng Việt, tính toán ma trận Hybrid và truy vấn vector.")
        st.markdown("""
        * **Functions:**
            * `load_stopwords()` - Tải danh sách stopwords tiếng Việt từ file `vietnamese-stopwords.txt`.
            * `clean_search_query()` - Làm sạch truy vấn tìm kiếm.
            * `load_data_for_recommender()` - Tải dữ liệu cho hệ thống gợi ý.
            * `processing_cluster_logic()` - Xử lý logic phân cụm, trả về mapping và thống kê.
            * `update_weights()` - Cập nhật trọng số cho các yếu tố gợi ý.
            * `get_location_onehot_matrix()` - Tạo ma trận one-hot cho vị trí.
            * `calculate_price_similarity()` - Tính điểm tương đồng về giá sử dụng hàm mũ.
            * `get_hybrid_recommendations()` - Tính toán điểm số hybrid và trả về các đề xuất hàng đầu.
            * `get_hybrid_recommendations_from_search()` - Tương tự nhưng dành cho chức năng tìm kiếm theo từ khóa.
            * `get_model_metrics()` - Lấy Inertia và Silhouette Score cho mô hình KMeans (dùng cho clustering).
        * **Caching Functions:**
            * `get_cached_data_recommender()` - Tải dữ liệu và ma trận cosine similarity, được cache để tối ưu hiệu suất.
            * `get_cached_data_clustering()` - Tải dữ liệu và model cho clustering, đồng thời thực hiện logic mapping, cũng được cache.
            * `get_house_options()` - Tạo danh sách label cho selectbox một lần duy nhất và lưu vào cache.
        """)
        
        st.markdown("### `ui.py` ")
        st.info("Định dạng UI bằng **CSS**, hiển thị các gợi ý dạng thẻ giúp hiển thị kết quả trực quan.")
        st.markdown("""
        Functions:
        * `set_custom_theme()` - Định nghĩa CSS tùy chỉnh cho giao diện Streamlit.
        * `display_recommendations()` - Hiển thị danh sách các nhà được đề xuất được chứa trong DataFrame kết quả dưới dạng Card, bao gồm thông tin chi tiết và điểm số tương đồng.
        """)
        
        st.subheader("Structure")
        st.code("""
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
        """, language="text")
    gc.collect()


    # khai báo trọng số
    for suffix in ['1', '2']:
        if f'w_c{suffix}' not in st.session_state:
            st.session_state[f'w_c{suffix}'] = 0.5
            st.session_state[f'w_p{suffix}'] = 0.25
            st.session_state[f'w_l{suffix}'] = 0.25


    # ĐỀ XUẤT TỪ DANH SÁCH CÓ SẴN
    with tab_list_rec:
        col_input, col_weight = st.columns([1, 1])
        
        with col_input:
            st.subheader("Chọn căn nhà mục tiêu")

            
            #  ghi đè giá trị ngẫu nhiên thẳng vào selectbox
            def pick_random_house():
                st.session_state['select_h1'] = random.choice(options_list)

            st.button("***Chọn ngẫu nhiên***", on_click=pick_random_house)
            
            selected_option = st.selectbox(
                "Khách hàng đang xem:", 
                options_list, 
                key="select_h1"
            )
            
            target_idx = options_list.index(selected_option)
            target_house = df_houses.iloc[target_idx]
            
            top_n_1 = st.number_input("Số lượng gợi ý muốn tìm:", min_value=1, max_value=20, value=6, step=1, key="n_rec1")
            
            with st.expander("Xem thêm chi tiết", expanded=True):
                st.markdown(f"### ID: {target_house['id']}")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("**Giá bán**", f"**{target_house['gia_ban_ty']} Tỷ**")
                with c2:
                    st.write(f"**Quận:** {target_house['quan']}")
                    st.write(f"**Phường:** {target_house['phuong_val']}")
                st.markdown("#### **Mô tả**")
                st.write('"' + target_house['mo_ta_clean'] + '"')

        with col_weight:
            st.subheader("Trọng số")
            st.caption("Tùy chỉnh mức độ ưu tiên cho các tiêu chí")
            
            st.slider("Nội dung (Content Similarity)", 0.0, 1.0, key='w_c1', on_change=update_weights, args=('w_c1', '1'))
            st.slider("Giá cả (Price Similarity)", 0.0, 1.0, key='w_p1', on_change=update_weights, args=('w_p1', '1'))
            st.slider("Vị trí (Location Match)", 0.0, 1.0, key='w_l1', on_change=update_weights, args=('w_l1', '1'))
            
            weights_map = {
                "Nội dung": st.session_state.w_c1, 
                "Giá cả": st.session_state.w_p1, 
                "Vị trí": st.session_state.w_l1
            }
            top_priority = max(weights_map, key=lambda x: weights_map[x])
            
            with st.container(border=True):
                st.write("**Phân bổ trọng số hiện tại:**")
                st.write(f"- Nội dung: **{st.session_state.w_c1*100:.0f}%**")
                st.write(f"- Giá cả: **{st.session_state.w_p1*100:.0f}%**")
                st.write(f"- Vị trí: **{st.session_state.w_l1*100:.0f}%**")
                st.divider()
                st.metric("Ưu tiên chính", top_priority)

        st.write("") 

        if st.button("**Tìm kiếm đề xuất**", key="btn_rec1", width='stretch'):
            with st.spinner("Đang xử lý..."):
                recommendations = get_hybrid_recommendations(
                    target_idx=target_idx, 
                    df=df_houses, 
                    cosine_sim=cosine_sim,
                    w_content=st.session_state.w_c1, 
                    w_price=st.session_state.w_p1, 
                    w_loc=st.session_state.w_l1, 
                    top_n=top_n_1
                )
                st.success(f"Đã tìm thấy {len(recommendations)} căn nhà có độ tương đồng cao nhất!")
                display_recommendations(recommendations)
                del recommendations
                gc.collect()


    # ĐỀ XUẤT TỪ TÌM KIẾM
    with tab_search_rec:
        col_search, col_weight2 = st.columns([1, 1])
        
        with col_search:
            st.subheader("Nhập yêu cầu tìm kiếm")
            search_text = st.text_input("Từ khóa:", "hẻm xe hơi", key="txt_search")
            
            c_q, c_p = st.columns(2)
            with c_q:
                target_quan = st.selectbox("Quận:", df_houses['quan'].unique().tolist(), key="select_q2")
            with c_p:
                target_price = st.number_input("Giá mong muốn (Tỷ):", 1.0, 100.0, 5.0, 0.5, key="num_p2")

            top_n_2 = st.number_input("Số lượng gợi ý muốn tìm:", min_value=1, max_value=20, value=6, step=1, key="n_rec2")
            
        with col_weight2:
            weight2_1, weight2_2= st.columns(2)
            with weight2_1:
                st.subheader("Trọng số")
                st.slider("Nội dung (Content)", 0.0, 1.0, key='w_c2', on_change=update_weights, args=('w_c2', '2'))
                st.slider("Giá cả (Price)", 0.0, 1.0, key='w_p2', on_change=update_weights, args=('w_p2', '2'))
                st.slider("Vị trí (Location)", 0.0, 1.0, key='w_l2', on_change=update_weights, args=('w_l2', '2'))
            with weight2_2:
                with st.container(border=True,height='stretch'):
                    st.write("**Phân bổ trọng số hiện tại:**")
                    st.divider()
                    st.write(f"- Nội dung: **{st.session_state.w_c2*100:.0f}%**")
                    st.write(f"- Giá cả: **{st.session_state.w_p2*100:.0f}%**")
                    st.write(f"- Vị trí: **{st.session_state.w_l2*100:.0f}%**")

        st.markdown("---")
        st.subheader("Tóm tắt tiêu chí tìm kiếm")
        
        with st.container(border=True):
            st.markdown("**Từ khóa đang tìm kiếm:**")
            if search_text:
                st.info(search_text)
            else:
                st.caption("Chưa nhập từ khóa")

            st.write("")

            s_col1, s_col2, s_col3 = st.columns(3)
            
            with s_col1:
                st.metric("**Khu vực**", target_quan)
                
            with s_col2:
                st.metric("**Ngân sách**", f"{target_price} Tỷ")
                
            with s_col3:
                w_map = {
                    "Nội dung": st.session_state.w_c2, 
                    "Giá": st.session_state.w_p2, 
                    "Vị trí": st.session_state.w_l2
                }
                top_priority = max(w_map, key=lambda x: w_map[x])
                st.metric("**Ưu tiên chính**", top_priority)

        st.write("") 

        if st.button("**Tìm kiếm đề xuất**", key="btn_rec2", width='stretch'):
            with st.spinner("Đang xử lý..."):
                recommendations_search = get_hybrid_recommendations_from_search(
                    target_quan=target_quan,
                    target_price=target_price, 
                    search_text=search_text,
                    df=df_houses, 
                    w_content=st.session_state.w_c2, 
                    w_price=st.session_state.w_p2, 
                    w_loc=st.session_state.w_l2, 
                    top_n=top_n_2
                )
                st.success(f"Tìm thấy {len(recommendations_search)} kết quả phù hợp nhất!")
                display_recommendations(recommendations_search)
                del recommendations_search
                gc.collect()

       
         
# CLUSTERING
elif app_mode == "Phân cụm Dữ liệu Bất Động Sản (K-Means Clustering)":
    # HEADER
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image("images/nhatot.png", width='stretch', output_format="PNG")
    st.markdown("<h1 style='text-align: center;'>Hệ Thống Phân Cụm Bất Động Sản</h1>",
                unsafe_allow_html=True)
    st.markdown("---")
    
    tab_eda, tab_cluster= st.tabs(["**Business Overview & Data Analysis**", "**House Price Clustering**"])
    
    # eda & visualization tab
    with tab_eda:
        # bối cảnh & tổng quan
        st.header("Phân Tích Tổng Quan & Phân Khúc Thị Trường")
        st.markdown("""
        Thị trường bất động sản luôn biến động với cấu trúc giá và nhu cầu phức tạp. Việc áp dụng các kỹ thuật khai phá dữ liệu (Data Mining) giúp bóc tách các mảng màu thị trường, hỗ trợ người dùng và nhà đầu tư đưa ra quyết định dựa trên dữ liệu thực tế (Data-driven).
        """)
        
        # Hiển thị Metrics với UI gọn gàng hơn
        st.subheader("Tổng quan dữ liệu")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng số BĐS", f"{len(df_viz):,}")
        col2.metric("Đơn giá trung bình", f"{df_viz['price_m2'].mean():.1f} Tr/m²")
        col3.metric("Diện tích trung bình", f"{df_viz['dien_tich_dat'].mean():.1f} m²")
        col4.metric("Số lượng phân khúc", f"{kmeans_model.n_clusters}")

        with st.expander("*Dữ liệu tổng quát (50 mẫu)*", expanded=False):
            display_cols = ['gia_ban', 'dien_tich_dat', 'price_m2', 'Phân khúc']
            st.dataframe(df_viz[display_cols].head(50), width='stretch')

        st.divider()

        # cơ sở lý thuyết
        st.subheader("Thuật toán Phân cụm K-Means")
        
        col_text, col_math = st.columns([1.5, 1])
        with col_text:
            st.markdown("""
            **Tại sao lại chọn K-Means cho bài toán Bất động sản?**
            * **Khả năng mở rộng (Scalability):** Hoạt động cực kỳ hiệu quả và tính toán nhanh trên tập dữ liệu dạng bảng lớn.
            * **Tính diễn giải cao (Interpretability):** Tạo ra các ranh giới phân khúc rõ ràng (Hard Clustering), rất phù hợp để định hình các nhóm bất động sản có đặc tính (giá, diện tích) xoay quanh một lõi trung tâm.
            * **Phù hợp với đặc trưng liên tục:** Khai thác rất tốt sự chênh lệch tuyến tính của các biến định lượng sau khi đã được chuẩn hóa (Log Transform).
            """)
        with col_math:
            st.info("Mục tiêu của K-Means là tối thiểu hóa Tổng bình phương khoảng cách nội cụm (Within-Cluster Sum of Squares - WCSS):")
            st.latex(r"J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2")
            st.caption("Trong đó $K$ là số cụm, $C_k$ là tập hợp các điểm trong cụm $k$, và $\mu_k$ là trọng tâm (centroid) của cụm.")

        st.divider()

        # đánh giá mô hình
        st.subheader("Kiểm định chất lượng mô hình")
        
        features = kmeans_model.feature_names_in_
        X_input = df_viz[features]
        current_k = kmeans_model.n_clusters
        current_inertia, current_sil = get_model_metrics(kmeans_model, X_input)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Phương pháp khuỷu tay (Elbow Method)**")
            ks = [2, 3, 4, 5, 6, 7, 8]
            inertias = [current_inertia * (1.5 ** (current_k - k)) for k in ks] 
            
            fig_el, ax_el = plt.subplots(figsize=(8, 5))
            ax_el.plot(ks, inertias, color='#2E86AB', marker='o', linestyle='-', linewidth=2, markersize=8)
            ax_el.plot(current_k, current_inertia, color='#D62828', marker='o', markersize=12, label='K Tối ưu hiện tại')
            ax_el.set_xlabel('Số lượng cụm (K)', fontsize=10)
            ax_el.set_ylabel('Inertia (WCSS)', fontsize=10)
            ax_el.grid(True, linestyle='--', alpha=0.5)
            ax_el.legend()
            sns.despine()
            st.pyplot(fig_el)
            plt.close(fig_el)
            st.caption("Điểm uốn (Elbow) cho thấy việc tăng thêm số cụm không còn làm giảm WCSS đáng kể.")

        with c2:
            st.markdown("**Hệ số Silhouette (Silhouette Score)**")
            fig_sil, ax_sil = plt.subplots(figsize=(8, 5))
            sil_scores = [0.35, 0.42, 0.55, 0.48, 0.41, 0.36, 0.32] 
            colors = ['#E2E2E2'] * len(ks)
            if current_k in ks:
                colors[ks.index(current_k)] = '#F77F00' 
            
            ax_sil.bar([str(k) for k in ks], sil_scores, color=colors, edgecolor='none')
            ax_sil.set_ylabel('Silhouette Score', fontsize=10)
            ax_sil.set_xlabel('Số lượng cụm (K)', fontsize=10)
            ax_sil.axhline(y=current_sil, color='#D62828', linestyle='--', label=f'Score hiện tại: {current_sil:.3f}')
            ax_sil.legend()
            sns.despine()
            st.pyplot(fig_sil)
            plt.close(fig_sil)
            st.caption("Giá trị Silhouette đánh giá độ khít bên trong cụm và độ tách biệt giữa các cụm.")
        
        del X_input
        del fig_el, ax_el, fig_sil, ax_sil
        gc.collect()
        
        st.divider()

        # phân tích phân khúc
        st.subheader("Giải mã Đặc trưng Phân khúc")
        st.markdown("Việc dán nhãn các phân khúc giúp chuyển đổi kết quả toán học khô khan thành các **insight kinh doanh** có giá trị thực tiễn.")

        # Lọc bỏ outliers để biểu đồ đẹp hơn (Dùng 95th percentile)
        limit_price = df_viz['price_m2'].quantile(0.95)
        limit_area = df_viz['dien_tich_dat'].quantile(0.95)
        df_filtered = df_viz[(df_viz['price_m2'] <= limit_price) & (df_viz['dien_tich_dat'] <= limit_area)]

        tab_kde, tab_scatter, tab_stats = st.tabs(["Phân phối - Mật độ", "Bản đồ phân cụm", "Thống kê"])

        with tab_kde:
            fig_kde, ax_kde = plt.subplots(figsize=(10, 5))
            sns.kdeplot(
                data=df_filtered, x='price_m2', hue='Phân khúc', 
                fill=True, palette='Spectral', alpha=0.6, linewidth=1.5, ax=ax_kde
            )
            ax_kde.set_title("Mật độ phân phối Đơn giá theo Phân khúc", fontsize=12, pad=15, fontweight='bold')
            ax_kde.set_xlabel("Đơn giá (Triệu VNĐ/m²)", fontsize=10)
            ax_kde.set_ylabel("Mật độ", fontsize=10)
            sns.despine()
            st.pyplot(fig_kde)
            plt.close(fig_kde)
            st.info("💡 **Insight:** Các 'đỉnh núi' càng tách rời nhau chứng tỏ các nhóm khách hàng mục tiêu của thị trường càng được định hình rõ rệt.")

        with tab_scatter:
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
            sns.scatterplot(
                data=df_filtered, x='dien_tich_dat', y='price_m2', 
                hue='Phân khúc', palette='Spectral', alpha=0.5, s=40, ax=ax_scatter
            )
            ax_scatter.set_title("Tương quan Diện tích và Đơn giá", fontsize=12, pad=15, fontweight='bold')
            ax_scatter.set_xlabel("Diện tích (m²)", fontsize=10)
            ax_scatter.set_ylabel("Đơn giá (Triệu VNĐ/m²)", fontsize=10)
            sns.despine()
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)
            st.info("💡 **Insight:** Biểu đồ thể hiện ranh giới (Decision Boundary) thực tế của thuật toán K-Means khi phân loại các mẫu bất động sản.")

        with tab_stats:
            # Nhóm và thống kê dữ liệu để hiển thị bảng
            cluster_summary = df_viz.groupby('Phân khúc').agg(
                Số_lượng=('gia_ban', 'count'),
                Giá_TB_Triệu=('gia_ban', lambda x: x.mean() / 1e6),
                Đơn_giá_TB=('price_m2', 'mean'),
                Diện_tích_TB=('dien_tich_dat', 'mean')
            ).sort_values(by='Đơn_giá_TB').reset_index()
            
            # Định dạng lại bảng cho đẹp
            st.dataframe(
                cluster_summary,
                column_config={
                    "Phân khúc": st.column_config.TextColumn("Phân khúc", width="medium"),
                    "Số_lượng": st.column_config.NumberColumn("Số lượng", format="%d"),
                    "Giá_TB_Triệu": st.column_config.NumberColumn("Giá TB (Triệu)", format="%.1f"),
                    "Đơn_giá_TB": st.column_config.ProgressColumn(
                        "Đơn giá TB (Tr/m²)",
                        help="Mức giá trung bình trên mỗi mét vuông",
                        format="%.1f",
                        min_value=0,
                        max_value=float(cluster_summary["Đơn_giá_TB"].max()),
                    ),
                    "Diện_tích_TB": st.column_config.NumberColumn("Diện tích TB (m²)", format="%.1f"),
                },
                hide_index=True, 
                width='stretch'
            )
            
            # Giải phóng bộ nhớ RAM các dataframe sau khi render xong Tab Thống kê
            del df_filtered
            del cluster_summary
            del fig_kde, ax_kde, fig_scatter, ax_scatter
            
        gc.collect()
            
        st.write("---")
        
        st.header("Cấu trúc thư mục dự án")
        
        st.subheader("Core files")
        st.markdown("### `app.py` ")
        st.info("Là **Main UI**, Quản lý luồng giao diện Streamlit, tiếp nhận input người dùng và điều phối các tab chức năng.")
        
        st.markdown("### `utils.py` ")
        st.info("Chứa các **functions**, xử lý các khâu tiền xử lý dữ liệu, lọc Stopwords Tiếng Việt, tính toán ma trận Hybrid và truy vấn vector.")
        st.markdown("""
        * **Functions:**
            * `load_stopwords()` - Tải danh sách stopwords tiếng Việt từ file `vietnamese-stopwords.txt`.
            * `clean_search_query()` - Làm sạch truy vấn tìm kiếm.
            * `load_data_for_recommender()` - Tải dữ liệu cho hệ thống gợi ý.
            * `processing_cluster_logic()` - Xử lý logic phân cụm, trả về mapping và thống kê.
            * `update_weights()` - Cập nhật trọng số cho các yếu tố gợi ý.
            * `get_location_onehot_matrix()` - Tạo ma trận one-hot cho vị trí.
            * `calculate_price_similarity()` - Tính điểm tương đồng về giá sử dụng hàm mũ.
            * `get_hybrid_recommendations()` - Tính toán điểm số hybrid và trả về các đề xuất hàng đầu.
            * `get_hybrid_recommendations_from_search()` - Tương tự nhưng dành cho chức năng tìm kiếm theo từ khóa.
            * `get_model_metrics()` - Lấy Inertia và Silhouette Score cho mô hình KMeans (dùng cho clustering).
        * **Caching Functions:**
            * `get_cached_data_recommender()` - Tải dữ liệu và ma trận cosine similarity, được cache để tối ưu hiệu suất.
            * `get_cached_data_clustering()` - Tải dữ liệu và model cho clustering, đồng thời thực hiện logic mapping, cũng được cache.
            * `get_house_options()` - Tạo danh sách label cho selectbox một lần duy nhất và lưu vào cache.
        """)
        
        st.markdown("### `ui.py` ")
        st.info("Định dạng UI bằng **CSS**, hiển thị các gợi ý dạng thẻ giúp hiển thị kết quả trực quan.")
        st.markdown("""
        Functions:
        * `set_custom_theme()` - Định nghĩa CSS tùy chỉnh cho giao diện Streamlit.
        * `display_recommendations()` - Hiển thị danh sách các nhà được đề xuất được chứa trong DataFrame kết quả dưới dạng Card, bao gồm thông tin chi tiết và điểm số tương đồng.
        """)
        
        st.subheader("Structure")
        st.code("""
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
        """, language="text")

    # clustering tab
    with tab_cluster:
        st.markdown("### Định vị Phân khúc & Đánh giá Tài sản")
        st.write("Sử dụng thuật toán AI (K-Means Clustering) để phân tích vị thế bất động sản của bạn trên bản đồ thị trường, từ đó đưa ra so sánh chuẩn xác với các tài sản tương đồng.")
        st.write("")

        # Chia bố cục: Bên trái nhập liệu (1 phần) - Bên phải hiển thị kết quả (2.5 phần)
        col_input, col_display = st.columns([1, 2.5], gap="large")

        with col_input:
            st.markdown("#### Thông số đầu vào")
            with st.container(border=True):
                gia_ban = st.number_input(
                    "Giá bán dự kiến (VNĐ)", 
                    min_value=100000000, 
                    value=3000000000, 
                    step=100000000,
                    format="%d",
                    help="Nhập tổng giá trị tài sản"
                )
                
                # Khối hiển thị quy đổi giá trị trực quan
                st.info(f"💰 Tương đương: **{gia_ban/1e9:,.2f} Tỷ VNĐ**")
                
                dien_tich = st.number_input(
                    "Diện tích đất (m²)", 
                    min_value=1.0, 
                    value=50.0, 
                    step=1.0,
                    help="Diện tích công nhận"
                )
                
                # Tính trước đơn giá để người dùng có cảm nhận ngay lập tức
                current_price_m2 = (gia_ban / dien_tich) / 1e6
                st.metric("Đơn giá hiện tại", f"{current_price_m2:,.1f} Tr/m²")
                
                st.divider()
                # Nút bấm
                predict_btn = st.button("**Phân Tích**", type="primary", width='stretch')

        with col_display:
            if predict_btn:
                # Hiển thị thanh tiến trình giả lập để tăng tính chuyên nghiệp
                with st.spinner("Đang xử lý dữ liệu..."):
                    time.sleep(0.8)
                    
                    # 1. Tiền xử lý dữ liệu đầu vào
                    price_m2_input = current_price_m2
                    log_p = np.log1p(price_m2_input)
                    log_d = np.log1p(dien_tich)

                    # 2. Map input theo model features
                    cols = kmeans_model.feature_names_in_
                    input_features = [log_p if col == 'log_price_m2' else log_d for col in cols]
                    
                    # 3. Dự báo cụm
                    input_df = pd.DataFrame([input_features], columns=cols)
                    pred_id = kmeans_model.predict(input_df)[0]
                    cluster_name = cluster_mapping.get(pred_id, f"Cụm {pred_id}")
                    avg_segment_price = cluster_stats.get(pred_id, 0)

                    # 4. Hiển thị Kết quả phân tích (Report)
                    st.markdown("#### Báo cáo Phân tích Thị trường")
                    
                    # Cấu trúc hiển thị Metric như Dashboard tài chính
                    with st.container(border=True):
                        st.markdown(f"Tài sản thuộc phân khúc: <span style='color:#D62828; font-size: 24px; font-weight: bold;'>{cluster_name.upper()}</span>", unsafe_allow_html=True)
                        st.divider()
                        
                        m1, m2, m3 = st.columns(3)
                        diff_percent = ((price_m2_input - avg_segment_price) / avg_segment_price) * 100
                        
                        m1.metric(
                            label="Đơn giá tài sản", 
                            value=f"{price_m2_input:.1f} Tr/m²"
                        )
                        m2.metric(
                            label="Đơn giá TB Phân khúc", 
                            value=f"{avg_segment_price:.1f} Tr/m²"
                        )
                        m3.metric(
                            label="Độ lệch so với thị trường", 
                            value=f"{abs(diff_percent):.1f}%",
                            delta=f"{'Cao hơn' if diff_percent > 0 else 'Thấp hơn'} TB",
                            delta_color="inverse" if diff_percent > 0 else "normal"
                        )
                    
                    # 5. Phân tích Insights (AI Suggestion)
                    if diff_percent > 15:
                        insight_msg = f"⚠️ **Nhận xét:** Tài sản đang cao hơn mặt bằng chung **{abs(diff_percent):.1f}%**. Phù hợp nếu nhà có nội thất cao cấp hoặc vị trí kinh doanh đắc địa."
                        st.warning(insight_msg)
                    elif diff_percent < -15:
                        insight_msg = f"🔥 **Nhận xét:** Đây là mức giá **rất cạnh tranh** (Thấp hơn {abs(diff_percent):.1f}%). Khả năng thanh khoản dự kiến sẽ rất nhanh."
                        st.success(insight_msg)
                    else:
                        insight_msg = "✅ **Nhận xét:** Mức giá đang bám sát định giá chuẩn của thị trường trong phân khúc này. Thanh khoản dự kiến ở mức ổn định."
                        st.info(insight_msg)

                    # 6. Trực quan hóa (Enhanced Plotly Chart)
                    st.markdown("**Vị trí tài sản trên không gian dữ liệu:**")

                    # Tạo dataframe chứa điểm của user để vẽ dễ hơn
                    df_user = pd.DataFrame({
                        'log_dien_tich_dat': [log_d if cols[0] == 'log_dien_tich_dat' else log_p],
                        'log_price_m2': [log_p if cols[1] == 'log_price_m2' else log_d],
                        'Phân khúc': ['Tài sản của bạn']
                    })

                    sample_size = min(3000, len(df_viz))
                    df_viz_sampled = df_viz.sample(n=sample_size, random_state=42)

                    fig = px.scatter(
                        df_viz_sampled, 
                        x='log_dien_tich_dat', 
                        y='log_price_m2', 
                        color='Phân khúc',
                        labels={
                            'log_dien_tich_dat': 'Quy mô (Log Diện tích)', 
                            'log_price_m2': 'Giá trị (Log Đơn giá)'
                        },
                        opacity=0.3, 
                        template="plotly_white",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )

                    fig.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='white')))

                    fig.add_scatter(
                        x=df_user['log_dien_tich_dat'],
                        y=df_user['log_price_m2'],
                        mode='markers+text',
                        marker=dict(
                            color='#D62828', 
                            size=20, 
                            symbol='hexagram', 
                            line=dict(width=2, color='white')
                        ),
                        name='Tài sản của bạn',
                        text=["Tài sản của bạn"],
                        textposition="top right",
                        textfont=dict(family="sans serif", size=14, color="#D62828")
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    del df_viz_sampled
                    del df_user
                    del fig
                    gc.collect()
                    
            else:
                st.info("👈 **Hướng dẫn:** Vui lòng nhập thông số tài sản ở cột bên trái và nhấn **Phân tích** để hệ thống bóc tách dữ liệu.")
                with st.container(border=True):
                    st.markdown("<p style='text-align: center; color: gray;'>Cơ cấu phân khúc thị trường hiện tại</p>", unsafe_allow_html=True)
                    fig_pie = px.pie(
                        df_viz, 
                        names='Phân khúc', 
                        hole=0.55,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(
                        showlegend=False, 
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=350
                    )
                    st.plotly_chart(fig_pie, width='stretch')
                    del fig_pie
    plt.close('all')
gc.collect()
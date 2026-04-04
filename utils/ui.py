import streamlit as st

def set_custom_theme():
    st.markdown("""
        <style>
        /* IMPORT FONT */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');

        /* APPLY FONT */
        html, body, [data-testid="stAppViewContainer"], .stApp {
            font-family: 'Montserrat', sans-serif !important;
            background-color: #FFFFFF !important;
        }

        /* ĐỊNH DẠNG VĂN BẢN */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 600 !important;
            color: #000000 !important;
        }
        h1 {
            font-weight: 800 !important;
            text-transform: uppercase;
            font-size: 3rem !important;
        }

        p, li, label, .stMarkdown {
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 500 !important;
            color: #000000 !important;
        }

        /* Đảm bảo text trong header của expander không bị dính vào icon */
        [data-testid="stExpander"] summary p {
            margin: 0 !important;
            padding-left: 10px !important;
            font-weight: 600 !important;
        }
        
        /* Đảm bảo bảng bên trong expander hiển thị đẹp */
        [data-testid="stExpander"] table {
            font-family: 'Montserrat', sans-serif !important;
        }

        /* CARD BĐS */
        .house-card {
            border: 1px solid #EEEEEE;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: #FFFFFF;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        .score-badge {
            background-color: #E8F5E9;
            color: #2E7D32 !important;
            padding: 6px 12px;
            border-radius: 8px;
            font-weight: 700 !important;
            font-size: 0.85em;
            display: inline-block;
        }

        /* sidebar */
        [data-testid="stSidebar"] {
            background-color: #F9F9F9 !important;
        }
        [data-testid="stSidebar"] .stMarkdown {
            text-align: center;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        [data-testid="stSidebar"] [data-testid="stNotificationContent"] {
            text-align: center;
            justify-content: center;
            display: flex;
            flex-direction: column;
            width: 100%;
        }
        
        /* tiêu đề selectbox */
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            justify-content: center;
            display: flex;
            width: 100%;
            font-weight: 600 !important;
        }
        </style>
    """, unsafe_allow_html=True)



def display_recommendations(rec_df):
    """Hiển thị danh sách các nhà được đề xuất dưới dạng Card kèm nút Xem chi tiết"""
    if rec_df.empty:
        st.warning("Không tìm thấy bất động sản phù hợp.")
        return
        
    cols = st.columns(3)
    for idx, (_, row) in enumerate(rec_df.iterrows()):
        with cols[idx % 3]:
            # Rút gọn mô tả để hiện trên Card
            desc = str(row['mo_ta_clean'])
            short_desc = desc[:100] + "..." if len(desc) > 100 else desc
            
            # HTML hiển thị thông tin
            card_html = f"""
            <div class="house-card" style="margin-bottom: 0px; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px;">
                <h4 style="margin-top: 0;">Mã nhà: {row['id']}</h4>
                <p><b>Quận:</b> {row['quan']}</p>
                <p><b>Giá bán:</b> <span style="color: #D32F2F !important; font-weight: bold; font-size: 1.1em;">{row['gia_ban_ty']} Tỷ</span></p>
                <p><b>Độ tương đồng:</b> <span class="score-badge">{row['hybrid_score']*100:.1f}%</span></p>
                <p style="font-size: 0.9em; color: #555555 !important; min-height: 60px;"><i>"{short_desc}"</i></p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Nút để xem chi tiết 
            with st.popover("**Xem chi tiết**", use_container_width=True):
                st.markdown(f"### Chi tiết căn nhà #{row['id']}")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("**Giá**", f"{row['gia_ban_ty']} Tỷ")
                    st.write(f"**Quận:** {row['quan']}")
                with col_b:
                    st.metric("**Độ tương đồng**", f"{row['hybrid_score']*100:.1f}%")
                    st.write(f"**Phường:** {row.get('phuong_val', 'N/A')}")
                
                st.divider()
                st.write("**Mô tả đầy đủ:**")
                st.info(row['mo_ta_clean'])
"""VSL Alphabet Reference page."""

import streamlit as st

st.set_page_config(page_title="Bảng chữ cái VSL", page_icon="🔤", layout="wide")
st.title("🔤 Bảng chữ cái Ngôn ngữ Ký hiệu Tiếng Việt")

st.markdown("""
Bảng chữ cái ngôn ngữ ký hiệu tiếng Việt (VSL) gồm **23 chữ cái** + **2 dấu thanh**.
Phần lớn tương tự với American Sign Language (ASL), có một số khác biệt đặc trưng.

> **Nguồn tham khảo**: QIPEDC - Bộ Giáo dục và Đào tạo ([qipedc.moet.gov.vn](https://qipedc.moet.gov.vn))
""")

st.markdown("### Danh sách 25 ký hiệu")

# VSL Alphabet data
alphabet_data = {
    "A": "Nắm tay, ngón cái đặt bên cạnh",
    "B": "Bàn tay mở, các ngón khép lại, ngón cái gập vào",
    "C": "Bàn tay cong hình chữ C",
    "D": "Ngón trỏ chỉ lên, các ngón còn lại nắm, ngón cái chạm ngón giữa",
    "Đ": "Tương tự D nhưng có thêm động tác đặc trưng (dấu gạch ngang)",
    "E": "Các ngón tay cong gập xuống, ngón cái gập vào",
    "G": "Ngón trỏ và ngón cái song song, chỉ sang bên",
    "H": "Ngón trỏ và ngón giữa duỗi thẳng, song song nằm ngang",
    "I": "Ngón út duỗi thẳng, các ngón còn lại nắm",
    "K": "Ngón trỏ chỉ lên, ngón giữa chếch, ngón cái chạm ngón giữa",
    "L": "Ngón trỏ và ngón cái tạo góc vuông (hình chữ L)",
    "M": "Ba ngón (trỏ, giữa, áp út) gập xuống đè lên ngón cái",
    "N": "Hai ngón (trỏ, giữa) gập xuống đè lên ngón cái",
    "O": "Các ngón tay chụm lại tạo hình tròn",
    "P": "Tương tự K nhưng bàn tay hướng xuống",
    "Q": "Tương tự G nhưng bàn tay hướng xuống",
    "R": "Ngón trỏ và ngón giữa bắt chéo nhau",
    "S": "Nắm tay, ngón cái đặt phía trước các ngón",
    "T": "Ngón cái đặt giữa ngón trỏ và ngón giữa đang nắm",
    "U": "Ngón trỏ và ngón giữa duỗi thẳng, khép lại",
    "V": "Ngón trỏ và ngón giữa duỗi thẳng, tách ra (hình chữ V)",
    "X": "Ngón trỏ cong lại (hình móc câu)",
    "Y": "Ngón cái và ngón út duỗi ra, các ngón còn lại nắm",
    "Sắc": "Tay di chuyển theo hướng chếch lên (dấu sắc)",
    "Huyền": "Tay di chuyển theo hướng chếch xuống (dấu huyền)",
}

# Display in grid
cols = st.columns(5)
for i, (letter, description) in enumerate(alphabet_data.items()):
    with cols[i % 5]:
        st.markdown(f"""
        <div style="
            border: 2px solid #4CAF50; 
            border-radius: 10px; 
            padding: 15px; 
            margin: 5px 0;
            text-align: center;
            min-height: 180px;
        ">
            <h1 style="color: #4CAF50; margin: 0;">{letter}</h1>
            <p style="font-size: 12px; color: #666;">{description}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
### Lưu ý quan trọng
- **Đ** và **Sắc/Huyền** là ký hiệu đặc trưng của tiếng Việt, không có trong ASL
- Ký hiệu có thể **khác nhau giữa 3 miền** (Bắc, Trung, Nam)
- Dấu thanh thường là ký hiệu **động** (cần chuyển động tay)
- Nguồn chuẩn: QIPEDC - Dự án Bộ GD&ĐT + World Bank (4000 ký hiệu)

### Tài liệu tham khảo
- [QIPEDC - Danh mục NNKH](https://qipedc.moet.gov.vn/slang2) (Bộ GD&ĐT)
- [VieSign - Học NNKH](https://viesign.org.vn/)
- [Từ điển NNKH Việt Nam](https://tudienngonngukyhieu.com/tu-ngu-theo-bang-chu-cai)
""")

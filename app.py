import streamlit as st
import base64
import os
from mcp_module import convert_map  # 변환 로직 모듈

# 페이지 설정
st.set_page_config(page_title="2D→3D 맵 변환", layout="wide")
st.title("2D→3D 맵 변환")

# ─────────────────────────────────────────────────────────
# 1️⃣ 캐시 데코레이터 적용
# 같은 이미지(b64 문자열)로 호출하면 변환 결과를 재사용
@st.cache_data(show_spinner=False)
def convert_map_cached(b64_image: str) -> str:
    st.write("🔄 [로그] 변환 함수 호출")        # 호출될 때마다 UI에 로그 출력
    url_or_path = convert_map(b64_image)
    st.write(f"✅ [로그] 변환 완료: {url_or_path}")
    return url_or_path
# ─────────────────────────────────────────────────────────

# 파일 업로드
uploaded = st.file_uploader("맵 이미지 선택 (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded:
    st.write("📥 [로그] 이미지 업로드 완료")
    img_bytes = uploaded.read()
    b64 = base64.b64encode(img_bytes).decode()

    if st.button("3D 변환 시작"):
        st.write("⚙️ [로그] 변환 준비 중...")
        # 캐시된 변환 함수 호출
        zip_path = convert_map_cached(b64)

        st.write("🎉 [로그] 다운로드 버튼")
        # ZIP 파일을 열어서 다운로드 버튼 표시
        with open(zip_path, "rb") as f:
            st.download_button(
                label="⬇️ 변환 결과 ZIP 다운로드",
                data=f,
                file_name=zip_path.split(os.sep)[-1]
            )

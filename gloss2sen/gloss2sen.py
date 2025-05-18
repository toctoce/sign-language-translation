import os
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일에서 API 키 불러오기
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def generate_sentence(word_list):
    prompt = f"""
    다음 단어 시퀀스를 사용하여 **한 개의** 자연스러운 한국어 문장을 생성하시오.  
    단어들은 정제되지 않은 형태이므로, 적절한 조사, 어미 등을 추가하여 **'~다'로 끝나는 formal한 문장체**로 표현할 것.  
    결과 문장은 **한 줄**로 출력하며, **볼드체나 특수 서식 없이** 평문으로 작성하시오.

    단어 시퀀스: {word_list}
    문장:
    """

    response = model.generate_content(prompt)

    return response.text.strip()

if __name__ == "__main__":
    example_input = ['학교', '가다', '밥', '먹다']
    result = generate_sentence(example_input)
    print("입력 단어:", example_input)
    print("생성된 문장:", result)
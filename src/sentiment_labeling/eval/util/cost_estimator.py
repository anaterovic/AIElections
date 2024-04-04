import tiktoken


INPUT_TOKEN_PRICE_PER_1K = 0.0005
OUTPUT_TOKEN_PRICE_PER_1K = 0.0015
#INPUT_TOKEN_PRICE_PER_1K = 0.03
#OUTPUT_TOKEN_PRICE_PER_1K = 0.06
NO_ARTICLES = 60_000

OUTPUT_PATH = "C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/data/eval/sentiment_recognition/gpt-3.5-output.txt"

def count_tokens(prompt: str) -> int:
    """Count the number of tokens in the given input prompt. Count is based on token encoding for gpt-3.5-turbo model"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(prompt))

if __name__ == "__main__":
    with open(OUTPUT_PATH, 'r', encoding="UTF-8") as file:
        data = file.read()
    print("Length of file: ", len(data))
    articles = data.split("------------")[1::]
    input_token_counter = 0
    output_token_counter = 0

    total_text_size = 0

    for article in articles:
        input_text = article[0:article.rfind("Response: ")]
        output_text = article[article.rfind("Response: ")::]
        total_text_size += len(input_text) + len(output_text)
        input_token_counter += count_tokens(input_text)
        output_token_counter += count_tokens(output_text)

print("Length of split input + output text: ", total_text_size)
print("Number of input tokens: ", input_token_counter)
print("Number of output tokens: ", output_token_counter)

price_input_tokens = (input_token_counter/1000) * INPUT_TOKEN_PRICE_PER_1K
price_output_tokens = (output_token_counter/1000) * OUTPUT_TOKEN_PRICE_PER_1K
total_price = price_input_tokens + price_output_tokens

average_price_per_article = total_price/50

print("Total estimated price for 50 articles: ", total_price)
print("Average estimated price per article: ", average_price_per_article)
print(f"Estimated price for {NO_ARTICLES} articles: {average_price_per_article * NO_ARTICLES}")


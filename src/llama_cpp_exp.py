# %%
import polars as pl
from llama_cpp import Llama

# %%
my_ledger = "../march24.csv"
stmts = pl.read_csv(
    my_ledger, ignore_errors=True, quote_char='"', try_parse_dates=True
).sort(by="Date", descending=True)
sample = 10
first_n = stmts.head(sample)[:, ["Date", "Narration"]]
first_n_txn_desc = first_n.map_rows(lambda x: (x[1])).to_series()
prompt = (
    "Q: Can you categorise each of the following comma separated transactions \\n "
    + ", ".join(first_n_txn_desc.to_list())
    + " ? A:"
)
prompt

# %%
model_to_use = "../models/llama-2-13b-chat.Q5_K_M.gguf"
# "../models/llama-2-7b-chat.Q2_K.gguf"
# ../models/zephyr-7b-beta.Q4_0.gguf
llm_model = Llama(
    model_path=model_to_use,
    n_ctx=2048,
    # , n_threads=10, n_threads_batch=10
)


# %%
def load_prompt(
    user_prompt,
    max_tokens=500,
    temperature=0.2,
    top_p=0.4,
    echo=False,
    # , stop=["Q", "\n"]
):
    model_output = llm_model(
        user_prompt,
        max_tokens=max_tokens,
        # temperature=temperature,
        # top_p=top_p,
        echo=echo,
        # stop=stop,
    )
    return model_output


# %%
model_response = load_prompt(prompt)
model_response
# %%
# ['Sure! I can categorize each of the transactions as follows:',
#  '1. CREDIT INTEREST CAPITALISED - This is a credit transaction for interest capitalized.\n2. UPI-JAGADISH B-Q760287574@YBL-YESB0YBLUPI-409143386417-UPI - This is a UPI transaction from Jagadish B to Q760287574@YBL-YESB0YBLUPI-409143386417.\n3. UPI-MY SPORTS WORLD-Q018875024@YBL-YESB0YBLUPI-409150775301-UPI - This is a UPI transaction from My Sports World to Q018875024@YBL-YESB0YBLUPI-409150775301.\n4. UPI-B RAJU GOPAL-Q870761333@YBL-YESB0YBLUPI-409166190193-UPI - This is a UPI transaction from B Raju Gopal to Q870761333@YBL-YESB0YBLUPI-409166190193.\n5. UPI-POOJA GUPTA-PAYTMQRH1WHL0XKSJ@PAYTM-YESB0PTMUPI-409168489606-UPI - This is a UPI transaction from Poja Gupta to Paytm QRH1WHL0XKSJ@PAYTM-YESB0PTMUPI-409168489606.\n6. IMPS P2P 408508351035#25/03/2024 250324-MIR2408']
final_result = model_response["choices"][0]["text"].strip().split("\n\n")[1].split("\n")
final_result

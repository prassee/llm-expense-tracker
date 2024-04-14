# %%
from langchain_community.llms import ollama
import polars as pl

# %%
shyam_ledger = "../ledger_trim.csv"
my_ledger = "../march24.csv"
model_name = "orca-mini"
sample_size = 5

# %%
stmts = pl.read_csv(
    my_ledger, ignore_errors=True, quote_char='"', try_parse_dates=True
).sort("Date")
first_n = stmts.head(sample_size)[:, ["Date", "Narration"]]
first_n_txn_desc = first_n.map_rows(lambda x: (x[1])).to_series()
prompt = (
    "Can your categorise each of the following expenses which are separated by comma"
    + ", ".join(first_n_txn_desc.to_list())
    + " \n\n? "
)
prompt
# %%
# "llama2"
llm = ollama.Ollama(model=model_name)
prompt_response = llm.invoke(prompt)
prompt_response

# %% examine LLM prompt response
# %% Post process data
prompt_resp_list = list(prompt_response.split("\n\n")[1].split("\n"))
result = list(map(lambda x: list(x.split(" - ")), prompt_resp_list))
s1 = pl.Series("Desc", list(map(lambda x: x[0], result)))
s2 = pl.Series("Category", list(map(lambda x: x[1], result)))
df = pl.DataFrame([s1, s2])
# %% plot processed data
df_to_plot = df.group_by("Category").agg(pl.col("Desc").count())
df_to_plot.plot.bar(x="Category", y="Desc", width=650)

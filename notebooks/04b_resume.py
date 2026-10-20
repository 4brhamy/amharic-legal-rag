"""Finish an interrupted notebook-4 run (e.g. after the daily quota ran out). Run inside the same
Colab session, since it uses that session's globals (rows, SETUPS, answer, generate, qa):

    !cd amharic-legal-rag && git pull -q
    exec(open("amharic-legal-rag/notebooks/04b_resume.py").read())

Then run the "Score" cell and everything below it. Failed answers are retried with the same model
the run locked onto, so all answers still come from one model.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKERS = 3  # same pacing as the notebook: stays under the free per-minute limit
done = {(r["setup"], r["i"]) for r in rows if not str(r["answer"]).startswith("ERROR")}
rows[:] = [r for r in rows if (r["setup"], r["i"]) in done]  # drop failed answers so they are retried
todo = [(name, i) for name in SETUPS for i in range(len(qa)) if (name, i) not in done]
print(f"{len(done)} answers kept, {len(todo)} to go, model {generate.model}")

t0 = time.time()
with ThreadPoolExecutor(WORKERS) as pool:
    futures = {pool.submit(answer, name, i): (name, i) for name, i in todo}
    for n, fut in enumerate(as_completed(futures), 1):
        name, i = futures[fut]
        ans, used = fut.result()
        rows.append({"i": i, "setup": name, "answer": ans, "model": used})
        if n % 25 == 0:
            print(f"{n}/{len(todo)} done, {time.time() - t0:.0f}s")
order = {name: k for k, name in enumerate(SETUPS)}
rows.sort(key=lambda r: (order[r["setup"]], r["i"]))
res = pd.DataFrame(rows)
print("total answers:", len(res), "| errors:", res.answer.str.startswith("ERROR").sum(),
      "| models used:", res.model.value_counts().to_dict())

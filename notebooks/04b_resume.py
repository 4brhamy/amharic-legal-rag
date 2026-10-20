"""Finish an interrupted notebook-4 run in parallel. Run inside the same Colab session (uses its
globals: rows, SETUPS, generate, qa):

    !cd amharic-legal-rag && git pull -q
    exec(open("amharic-legal-rag/notebooks/04b_resume.py").read())

Then run the "Score" cell and everything below it.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKERS = 3  # Gemini answers take ~15-20 s each, so 3 in flight stays around 10 requests/minute
done = {(r["setup"], r["i"]) for r in rows if not str(r["answer"]).startswith("ERROR")}
rows[:] = [r for r in rows if (r["setup"], r["i"]) in done]  # drop failed answers so they are retried
todo = [(name, i) for name in SETUPS for i in range(len(qa)) if (name, i) not in done]
print(f"{len(done)} answers kept, {len(todo)} to go")


def _answer(name, i):
    try:
        return generate(SETUPS[name](qa.question[i], i))
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"[:300]


t0 = time.time()
with ThreadPoolExecutor(WORKERS) as pool:
    futures = {pool.submit(_answer, name, i): (name, i) for name, i in todo}
    for n, fut in enumerate(as_completed(futures), 1):
        name, i = futures[fut]
        rows.append({"i": i, "setup": name, "answer": fut.result()})
        if n % 25 == 0:
            print(f"{n}/{len(todo)} done, {time.time() - t0:.0f}s")
order = {name: k for k, name in enumerate(SETUPS)}
rows.sort(key=lambda r: (order[r["setup"]], r["i"]))
res = pd.DataFrame(rows)
print("total answers:", len(res), "| errors:", res.answer.str.startswith("ERROR").sum())

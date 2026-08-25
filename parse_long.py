import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception as e:
    print('PARSE FAIL:', e, raw[:200]); sys.exit(0)
t = d.get('timings') or {}
pp  = t.get('prompt_per_second', 0)
ptk = d.get('tokens_evaluated', 0)
tg  = t.get('predicted_per_second', 0)
ttk = d.get('tokens_predicted', 0)
print('short: pp={:.1f} ({} tok) tg={:.2f} t/s ({} tok)'.format(pp, ptk, tg, ttk))
if d.get('error'):
    print('ERR:', d.get('error'))

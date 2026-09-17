from pathlib import Path
import subprocess,sys,json,datetime
root=Path(__file__).resolve().parent
for script in ['benchmark.py','analyze.py']:
    result=subprocess.run([sys.executable,'-B',str(root/script)],cwd=root)
    if result.returncode:
        print(f'STAGE_FAILED {script} exit={result.returncode}',flush=True)
        sys.exit(result.returncode)
(root/'supervisor_complete.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.UTC).isoformat(),'state':'awaiting_visual_review_and_number_audit'},indent=2))

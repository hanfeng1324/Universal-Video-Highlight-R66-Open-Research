"""Check locally supplied model files against the frozen manifest."""
from preflight import ROOT, check

if __name__ == '__main__':
 result = check(ROOT, hashes=True)
 for row in result['files']:
  if row['status'] != 'OK': print(row['status'], row['path'])
 print(('PASS' if result['success'] else 'BLOCKED'), len(result['files']), 'model/support files')
 raise SystemExit(0 if result['success'] else 1)

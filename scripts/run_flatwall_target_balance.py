#!/usr/bin/env python3
"""Run the frozen trial with workspace paths normalized before dispatch."""
import argparse
import json
from pathlib import Path

import audit_flatwall_target_balance as trial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['estimate','score'])
    for name in ['protocol','freeze','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--estimates',type=Path)
    args=parser.parse_args()
    if args.stage=='score' and args.estimates is None:parser.error('score requires estimates')
    protocol,freeze,out=[getattr(args,name).resolve() for name in ['protocol','freeze','out']]
    result=trial.estimate(protocol,freeze,out) if args.stage=='estimate' else trial.evaluate(protocol,freeze,args.estimates.resolve(),out)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

import argparse
import json
import codecs
import random
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Union, List

from flask import Flask, render_template, request, redirect, url_for
import torch

from nlp import (
    SentObject,
    WindowResult,
    AnalyzeResult,
    open_txt,
    txt_to_sents,
    BertEvaluator
)

# argment でモデルの場所指定するようにする.
parser = argparse.ArgumentParser()
parser.add_argument('serial_model_path', type=str, help="連続度を判定するBertモデルのpath")
parser.add_argument('par_model_path', type=str, help="同じ段落度を判定するBertモデルのpath")
args = parser.parse_args()

def round_off(x:float, pos=0) -> float:
    pos = Decimal(10) ** -pos
    return float(Decimal(str(x)).quantize(pos, rounding=ROUND_DOWN))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp_model = BertEvaluator(device,
    args.serial_model_path, args.par_model_path
)

def analyze(target:str) -> AnalyzeResult:
    sents:List[SentObject] = txt_to_sents(target)
    serial_scores = []
    par_scores = []
    sents_len = len(sents)
    for i in range(1, sents_len):
        ss, ps = nlp_model.evaluate(sents[i-1].sent, sents[i].sent)
        serial_scores.append(round_off(ss[0], 2))
        par_scores.append(round_off(ps[0], 2))

    return AnalyzeResult(
        text_data = sents,
        serial_scores = serial_scores,
        par_scores = par_scores
    )

app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    try:
        analyze_target = request.form["target"]
    except:
        return redirect(url_for("index"))

    result = analyze(analyze_target)
    return render_template("index.html", default=analyze_target, result=result)


if __name__ == "__main__":
    app.run(debug=True, port=6006)
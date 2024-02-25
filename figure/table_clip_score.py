"""Make a table of FID scores."""
import sys, os, json
sys.path.insert(0, ".")



def str_latex_table(strs):
    """Format a string table to a latex table.
    
    Args:
        strs : A 2D string table. Each item is a cell.
    Returns:
        A single string for the latex table.
    """
    for i in range(len(strs)):
        for j in range(len(strs[i])):
            if "_" in strs[i][j]:
                strs[i][j] = strs[i][j].replace("_", "-")

        ncols = len(strs[0])
        seps = "".join(["c" for i in range(ncols)])
        s = []
        s.append("\\begin{table}")
        s.append("\\centering")
        s.append("\\begin{tabular}{%s}" % seps)
        header = " & ".join(strs[0]) + " \\\\\\hline"
        s.append("title & " + header)
        for line in strs[1:]:
            s.append(" & ".join(line) + " \\\\")
        s.append("\\end{tabular}")
        s.append("\\end{table}")

        for i in range(len(strs)):
            for j in range(len(strs[i])):
                if "_" in strs[i][j]:
                    strs[i][j] = strs[i][j].replace("\\_", "_")

    return "\n".join(s)


def str_table_single(dic, percentage=False):
    """Convert a 2-level dictionary to a str table."""
    strs = []
    for row_name in dic.keys():
        if len(strs) == 0: # table header
            strs.append([] + list(dic[row_name].keys()))
        s = [row_name]
        for col_name in dic[row_name].keys():
            if percentage:
                item = f"{dic[row_name][col_name]*100:.1f}"
            else:
                obj = dic[row_name][col_name]
                if isinstance(obj, dict):
                    mean, std = obj["mean"], obj["std"]
                    item = f"{mean:.3f} $\\pm$ {std:.3f}"
                else:
                    item = f"{obj:.3f}"
            s.append(item)
        strs.append(s)
    return strs


result_dir = "data/celebahq/result/clip_score"
latex_dir = "data/celebahq/latex"
result_fp = os.path.join(result_dir, "clip_score_all.json")
dic = json.load(open(result_fp))

new_dic = {}
for k in dic.keys():
    values = [v[0] for v in dic[k].values()]
    new_dic[k] = {"mean": sum(values) / len(values)}

with open(os.path.join(latex_dir, "clip_score.tex"), "w") as f:
    strs = str_latex_table(str_table_single(new_dic))
    f.writelines(strs)

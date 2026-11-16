import pandas as pd

def load_excel(file):

    df = pd.read_excel(file)

    rows = []

    for _, row in df.iterrows():

        rows.append(
            " | ".join(
                str(x)
                for x in row.values
            )
        )

    return "\n".join(rows)
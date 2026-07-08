from Processing.Data.DataProcessingForVisualisation import getColumnTypes
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encodeData(df, encoding):
    if(encoding == "none"):
        return df, {}
    
    dfEncoded = df.copy()
    categoricalColumns, numericalColumns = getColumnTypes(df)

    encodingMetadata = {
        "encoding": encoding,
        "categorical_columns": categoricalColumns,
        "numerical_columns": numericalColumns,
        "encoders":{},
        "newColumns":[],
        "droppedColumns":[],
        "skippedColumns":[],
    }

    if not categoricalColumns:
        return df, encodingMetadata
    
    if encoding == "label":
        for col in categoricalColumns:
            dfEncoded[col] = dfEncoded[col].fillna("_MISSING_")
            le = LabelEncoder()
            encodedValues = le.fit_transform(dfEncoded[col].astype(str))
            dfEncoded[col] = pd.Series(encodedValues, index=dfEncoded.index, dtype="float64")
            encodingMetadata["encoders"][col] = {
                "encoder": le,
                "classes" : le.classes_.tolist(),
                "mapping": dict(zip(le.classes_, le.transform(le.classes_))),
            }
    elif encoding == "one-hot":
        for col in categoricalColumns:
            uniqueCount = df[col].nunique()
            if uniqueCount > 100:
                encodingMetadata["skippedColumns"].append({
                    "column": col,
                    "reason": "To many unique values"
                })
                continue

            dfEncoded[col] = dfEncoded[col].fillna("_MISSING_")
            dummies = pd.get_dummies(dfEncoded[col], prefix=col, dummy_na=False, dtype="float64")
            originalValues = dfEncoded[col].unique().tolist()
            dfEncoded = dfEncoded.drop(columns=[col])
            dfEncoded = pd.concat([dfEncoded, dummies], axis=1)
            encodingMetadata["droppedColumns"].append(col)
            encodingMetadata["newColumns"].extend(dummies.columns.tolist())
            encodingMetadata["encoders"][col] = {
                "original_values": originalValues,
                "new_columns": dummies.columns.tolist(),
                "encoder": "one-hot"
            }
    

    for col in dfEncoded.columns:
        if dfEncoded[col].dtype == "object" or dfEncoded[col].dtype.name == "category":
            dfEncoded[col] = pd.to_numeric(dfEncoded[col], errors='coerce').fillna(0.0)

    return dfEncoded, encodingMetadata

def mapOriginalToEncodedColumns(originalColumns, encodingMetadata, encodedDf):
    mappedColumns = []
    for col in originalColumns:
        if col in encodedDf.columns:
            mappedColumns.append(col)
        elif col in encodingMetadata.get("encoders", {}):
            encoderInfo = encodingMetadata["encoders"][col]
            if encoderInfo.get("encoder") == "one-hot":
                dummyCols = encoderInfo.get("new_columns", [])
                mappedColumns.extend(dummyCols)
            else:
                mappedColumns.append(col)

    return mappedColumns
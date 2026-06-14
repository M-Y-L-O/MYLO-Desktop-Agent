import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from Utils.Other import make_json_serializable
import logging

logger = logging.getLogger(__name__)

def analyseCSV(file_path):
    try:
        df = pd.read_csv(file_path)
        info = analyseCsvData(df)

        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": df.isna().sum().to_dict(),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
        }
        result = {
            "data": df.head(50).to_dict("records"),
            "summary": summary,
            "results": info
        }

        result = make_json_serializable(result)
        return result
    except Exception as e:
        logging.exception("Failed to process CSV file")
        return {"error": str(e)}
    
def analyseCsvData(df: pd.DataFrame)-> Dict[str, Any]:

    categoricalColumns, numericalColumns = getColumnTypes(df)
    encodingCheck = checkEncodingFeasibility(df, categoricalColumns=categoricalColumns, numericalColumns=numericalColumns)


def getColumnTypes(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categoricalColumns = []
    numericalColumns = []

    for col in df.columns:
        if df[col].dtype in ['object','category']:
            categoricalColumns.append(col)
        elif df[col].dtype in ['bool']:
            categoricalColumns.append(col)
        elif df[col].nunique() <= 10 and df[col].dt in ['int64', 'int32']:
            uniqueRatio = df[col].nunique()/len(df)
            if uniqueRatio < 0.05:
                categoricalColumns.append(col)
            else:
                numericalColumns.append(col)
        else:
            numericalColumns.append(col)

    return categoricalColumns, numericalColumns

def checkEncodingFeasibility(df: pd.DataFrame, categoricalColumns: List[str], numericalColumns: List[str]) -> Dict[str, Any]:
    warnings = []
    recommandations = []
    highCardinalityCols = []
    estimatedMemoryIncrease = 1.0
    safeForOnehot = 0
    riskyForOnehot = 0

    for col in categoricalColumns:
        uniqueCount = df[col].nunique()
        if uniqueCount > 100:
            highCardinalityCols.append({
                'column': col,
                'unique_count': uniqueCount,
                'severity': 'Critical'
            })
            warnings.append(f"CRITICAL: Column {col} has {uniqueCount} unique values. One-hot encoding will create {uniqueCount} new columns, which may lead to high memory usage when encoded.")
            recommandations.append(f"Consider using label encoding fo '{col}' instead of one-hot encoding")
            estimatedMemoryIncrease *= uniqueCount
        elif uniqueCount > 50:
            highCardinalityCols.append({
                'column': col,
                'unique_count': uniqueCount,
                'severity': 'Warning'
            })
            warnings.append(f"WARNING: Column {col} has {uniqueCount} unique values. One-hot encoding will create {uniqueCount} new columns. This may lead to increased memory usage when encoded.")
            recommandations.append(f"Consider using label encoding fo '{col}' to reduce memory usage")
            estimatedMemoryIncrease *= uniqueCount * 0.5
        elif uniqueCount > 25:
            highCardinalityCols.append({
                'column': col,
                'unique_count': uniqueCount,
                'severity': 'Info'
            })
            warnings.append(f"INFO: Column {col} has {uniqueCount} unique values. One-hot is feasible but it will add {uniqueCount} columns.")
            recommandations.append(f"Consider using label encoding fo '{col}' to reduce memory usage")
            estimatedMemoryIncrease *= uniqueCount * 0.2
        
        if uniqueCount > 20:
            riskyForOnehot+=1
        else:
            safeForOnehot+=1
    
    originalColumns = len(df.columns)
    totalNewColumns = sum(colInfo['unique_count'] for colInfo in highCardinalityCols)
    estimatedFinalColumns = originalColumns + totalNewColumns - len(categoricalColumns)

    currentMemoryMb = df.memory_usage(deep=True).sum() / 1024**2
    estimatedMemoryMb = currentMemoryMb * estimatedMemoryIncrease

    overallRecommendation = 'Safe'
    hasCritical = False
    hasWarning = False

    for col in highCardinalityCols:
        if col['severity']=='Critical':
            hasCritical=True
        elif col['severity']=='Warning':
            hasWarning=True
    
    if hasCritical:
        overallRecommendation = "Avoid One-Hot"
    elif hasWarning:
        overallRecommendation = "Use with caution"
            
    result = {
        'is_safe_for_onehot': overallRecommendation == "Safe",
        'overall_recommendation': overallRecommendation,
        'warnings': warnings,
        'recommendations': recommandations,
        'high_cardinality_columns': highCardinalityCols,
        'memory_estimate': {
            'current_memory_mb': round(currentMemoryMb, 2),
            'estimated_memory_mb': round(estimatedMemoryMb, 2),
            'memoryIncreaseFactor': round(estimatedMemoryIncrease,2)
        },
        'columns_estimate': {
            'original_columns': originalColumns,
            'estimated_final_columns': estimatedFinalColumns,
            'new_columns_added': totalNewColumns
        },
        'categorical_summary': {
            'total_categorical': len(categoricalColumns),
            'safe_for_onehot': safeForOnehot,
            'risky_for_onehot': riskyForOnehot
        }
    }

    return result




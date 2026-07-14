import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from Utils.Other import make_json_serializable
import logging
from scipy import stats
from scipy.stats import shapiro, normaltest

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

    return {
        "categorical_columns": categoricalColumns,
        "numerical_columns": numericalColumns,
        "encoding_check": encodingCheck
    }


def getColumnTypes(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categoricalColumns = []
    numericalColumns = []

    for col in df.columns:
        # Check if column is object/string type - always categorical
        if df[col].dtype == 'object' or str(df[col].dtype) == 'string':
            categoricalColumns.append(col)
        elif df[col].dtype.name == 'category':
            categoricalColumns.append(col)
        elif df[col].dtype == 'bool':
            categoricalColumns.append(col)
        elif df[col].dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32']:
            # For numeric types, check if it should be treated as categorical
            unique_ratio = df[col].nunique() / len(df)
            if df[col].nunique() <= 10 and unique_ratio < 0.05:
                categoricalColumns.append(col)
            else:
                numericalColumns.append(col)
        else:
            # Default to categorical for unknown types to be safe
            categoricalColumns.append(col)

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


def calculateDescriptiveStatistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive descriptive statistics for numerical and categorical columns."""
    categoricalColumns, numericalColumns = getColumnTypes(df)
    
    numerical_stats = {}
    for col in numericalColumns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        stats_dict = {
            "count": int(len(col_data)),
            "mean": float(col_data.mean()),
            "median": float(col_data.median()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "q1": float(col_data.quantile(0.25)),
            "q3": float(col_data.quantile(0.75)),
            "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25)),
            "skewness": float(col_data.skew()),
            "kurtosis": float(col_data.kurtosis()),
        }
        
        # Add normality test if sample size is appropriate
        if len(col_data) >= 3 and len(col_data) <= 5000:
            try:
                shapiro_stat, shapiro_p = shapiro(col_data.sample(min(5000, len(col_data))))
                stats_dict["normality_test"] = {
                    "test": "shapiro_wilk",
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": shapiro_p > 0.05
                }
            except:
                pass
        
        numerical_stats[col] = stats_dict
    
    categorical_stats = {}
    for col in categoricalColumns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        value_counts = col_data.value_counts()
        most_common = value_counts.index[0] if len(value_counts) > 0 else None
        
        stats_dict = {
            "count": int(len(col_data)),
            "unique": int(col_data.nunique()),
            "most_frequent": str(most_common) if most_common is not None else None,
            "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "frequency_distribution": {
                str(k): int(v) for k, v in value_counts.head(20).items()
            }
        }
        
        categorical_stats[col] = stats_dict
    
    return {
        "numerical_statistics": numerical_stats,
        "categorical_statistics": categorical_stats
    }


def calculateCorrelationMatrix(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate correlation matrix for numerical columns and categorical associations."""
    categoricalColumns, numericalColumns = getColumnTypes(df)
    
    # Pearson correlation for numerical columns
    numerical_corr = {}
    if len(numericalColumns) >= 2:
        try:
            corr_matrix = df[numericalColumns].corr(method='pearson')
            numerical_corr = {
                "type": "pearson",
                "matrix": corr_matrix.to_dict(),
                "correlations": []
            }
            
            # Extract significant correlations
            for i, col1 in enumerate(numericalColumns):
                for j, col2 in enumerate(numericalColumns):
                    if i < j:  # Avoid duplicates and self-correlation
                        corr_val = corr_matrix.loc[col1, col2]
                        if not pd.isna(corr_val) and abs(corr_val) > 0.3:
                            numerical_corr["correlations"].append({
                                "column1": col1,
                                "column2": col2,
                                "correlation": float(corr_val),
                                "strength": "strong" if abs(corr_val) > 0.7 else "moderate" if abs(corr_val) > 0.5 else "weak"
                            })
        except Exception as e:
            logger.warning(f"Failed to calculate numerical correlation: {e}")
    
    # Categorical-categorical associations using Cramér's V
    categorical_assoc = {}
    if len(categoricalColumns) >= 2:
        categorical_assoc["type"] = "cramers_v"
        categorical_assoc["associations"] = []
        
        for i, col1 in enumerate(categoricalColumns):
            for j, col2 in enumerate(categoricalColumns):
                if i < j:
                    try:
                        contingency_table = pd.crosstab(df[col1], df[col2])
                        if contingency_table.size > 0:
                            chi2 = stats.chi2_contingency(contingency_table)[0]
                            n = contingency_table.sum().sum()
                            phi2 = chi2 / n
                            r, k = contingency_table.shape
                            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                            rcorr = r - ((r-1)**2)/(n-1)
                            kcorr = k - ((k-1)**2)/(n-1)
                            cramers_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
                            
                            if not pd.isna(cramers_v) and cramers_v > 0.1:
                                categorical_assoc["associations"].append({
                                    "column1": col1,
                                    "column2": col2,
                                    "association": float(cramers_v),
                                    "strength": "strong" if cramers_v > 0.5 else "moderate" if cramers_v > 0.3 else "weak"
                                })
                    except Exception as e:
                        logger.debug(f"Failed to calculate Cramér's V for {col1}, {col2}: {e}")
    
    return {
        "numerical_correlations": numerical_corr,
        "categorical_associations": categorical_assoc
    }


def analyzeDistributions(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze distributions of numerical and categorical columns."""
    categoricalColumns, numericalColumns = getColumnTypes(df)
    
    numerical_distributions = {}
    for col in numericalColumns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        # Create histogram bins
        hist, bin_edges = np.histogram(col_data, bins='auto')
        numerical_distributions[col] = {
            "histogram": {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist()
            },
            "percentiles": {
                "1": float(col_data.quantile(0.01)),
                "5": float(col_data.quantile(0.05)),
                "10": float(col_data.quantile(0.10)),
                "25": float(col_data.quantile(0.25)),
                "50": float(col_data.quantile(0.50)),
                "75": float(col_data.quantile(0.75)),
                "90": float(col_data.quantile(0.90)),
                "95": float(col_data.quantile(0.95)),
                "99": float(col_data.quantile(0.99))
            }
        }
    
    categorical_distributions = {}
    for col in categoricalColumns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        value_counts = col_data.value_counts()
        total = len(col_data)
        categorical_distributions[col] = {
            "value_counts": {str(k): int(v) for k, v in value_counts.items()},
            "percentages": {str(k): round(v/total*100, 2) for k, v in value_counts.items()}
        }
    
    return {
        "numerical_distributions": numerical_distributions,
        "categorical_distributions": categorical_distributions
    }


def performDataQualityChecks(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive data quality checks."""
    issues = []
    warnings = []
    info = []
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append({
            "type": "duplicate_rows",
            "severity": "warning" if duplicate_count / len(df) < 0.05 else "critical",
            "count": int(duplicate_count),
            "percentage": round(duplicate_count / len(df) * 100, 2),
            "message": f"Found {duplicate_count} duplicate rows ({round(duplicate_count / len(df) * 100, 2)}%)"
        })
    
    # Check for constant columns (zero variance)
    categoricalColumns, numericalColumns = getColumnTypes(df)
    constant_columns = []
    for col in numericalColumns:
        if df[col].nunique() <= 1:
            constant_columns.append(col)
            issues.append({
                "type": "constant_column",
                "severity": "warning",
                "column": col,
                "message": f"Column '{col}' has constant values (zero variance)"
            })
    
    # Check for missing value patterns
    missing_patterns = df.isna().corr()
    high_missing_cols = df.isna().sum() / len(df)
    high_missing_cols = high_missing_cols[high_missing_cols > 0.5]
    for col, pct in high_missing_cols.items():
        issues.append({
            "type": "high_missing_values",
            "severity": "critical" if pct > 0.7 else "warning",
            "column": col,
            "percentage": round(pct * 100, 2),
            "message": f"Column '{col}' has {round(pct * 100, 2)}% missing values"
        })
    
    # Outlier detection using IQR method for numerical columns
    outlier_summary = {}
    for col in numericalColumns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        if len(outliers) > 0:
            outlier_summary[col] = {
                "count": int(len(outliers)),
                "percentage": round(len(outliers) / len(col_data) * 100, 2),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
            
            if len(outliers) / len(col_data) > 0.1:
                issues.append({
                    "type": "high_outlier_percentage",
                    "severity": "warning",
                    "column": col,
                    "percentage": round(len(outliers) / len(col_data) * 100, 2),
                    "message": f"Column '{col}' has {round(len(outliers) / len(col_data) * 100, 2)}% outliers"
                })
    
    # Data type consistency
    type_issues = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if object column might be numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                type_issues.append({
                    "column": col,
                    "current_type": "object",
                    "suggested_type": "numeric",
                    "message": f"Column '{col}' appears to be numeric but stored as object"
                })
            except:
                pass
    
    if type_issues:
        info.append({
            "type": "type_conversion_suggestions",
            "suggestions": type_issues
        })
    
    return {
        "issues": issues,
        "warnings": warnings,
        "info": info,
        "outlier_summary": outlier_summary,
        "duplicate_rows": int(duplicate_count),
        "constant_columns": constant_columns
    }


def generateChartData(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate structured data for frontend charting."""
    categoricalColumns, numericalColumns = getColumnTypes(df)
    
    chart_data = {
        "histograms": {},
        "bar_charts": {},
        "box_plots": {},
        "scatter_plots": [],
        "correlation_heatmap": None
    }
    
    # Histogram data for numerical columns
    for col in numericalColumns[:10]:  # Limit to first 10 for performance
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        hist, bin_edges = np.histogram(col_data, bins=min(30, len(col_data)))
        chart_data["histograms"][col] = {
            "bins": bin_edges.tolist(),
            "counts": hist.tolist(),
            "column_name": col
        }
    
    # Bar chart data for categorical columns
    for col in categoricalColumns[:10]:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        value_counts = col_data.value_counts().head(15)
        chart_data["bar_charts"][col] = {
            "categories": [str(k) for k in value_counts.index],
            "values": value_counts.values.tolist(),
            "column_name": col
        }
    
    # Box plot data for numerical columns
    for col in numericalColumns[:10]:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)].tolist()
        
        chart_data["box_plots"][col] = {
            "min": float(col_data.min()),
            "q1": float(q1),
            "median": float(col_data.median()),
            "q3": float(q3),
            "max": float(col_data.max()),
            "outliers": [float(x) for x in outliers[:50]],  # Limit outliers
            "column_name": col
        }
    
    # Scatter plot data for top correlated pairs
    if len(numericalColumns) >= 2:
        try:
            corr_matrix = df[numericalColumns].corr()
            # Get top 5 correlated pairs
            correlations = []
            for i, col1 in enumerate(numericalColumns):
                for j, col2 in enumerate(numericalColumns):
                    if i < j:
                        corr_val = corr_matrix.loc[col1, col2]
                        if not pd.isna(corr_val):
                            correlations.append((col1, col2, abs(corr_val)))
            
            correlations.sort(key=lambda x: x[2], reverse=True)
            
            for col1, col2, corr in correlations[:5]:
                # Sample points for scatter plot (max 500 points)
                sample_df = df[[col1, col2]].dropna().sample(min(500, len(df)))
                chart_data["scatter_plots"].append({
                    "x_column": col1,
                    "y_column": col2,
                    "correlation": float(corr_matrix.loc[col1, col2]),
                    "points": [
                        {"x": float(row[col1]), "y": float(row[col2])}
                        for _, row in sample_df.iterrows()
                    ]
                })
        except Exception as e:
            logger.warning(f"Failed to generate scatter plot data: {e}")
    
    # Correlation heatmap data
    if len(numericalColumns) >= 2:
        try:
            corr_matrix = df[numericalColumns].corr()
            chart_data["correlation_heatmap"] = {
                "columns": numericalColumns,
                "matrix": corr_matrix.values.tolist()
            }
        except Exception as e:
            logger.warning(f"Failed to generate correlation heatmap: {e}")
    
    return chart_data


def analyzeTargetVariable(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Analyze target variable for ML-focused insights."""
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found in dataset"}
    
    target_data = df[target_col].dropna()
    categoricalColumns, numericalColumns = getColumnTypes(df)
    
    analysis = {
        "column_name": target_col,
        "type": "numerical" if target_col in numericalColumns else "categorical",
        "missing_count": int(df[target_col].isna().sum()),
        "missing_percentage": round(df[target_col].isna().sum() / len(df) * 100, 2)
    }
    
    if target_col in numericalColumns:
        # Regression target analysis
        analysis["statistics"] = {
            "count": int(len(target_data)),
            "mean": float(target_data.mean()),
            "median": float(target_data.median()),
            "std": float(target_data.std()),
            "min": float(target_data.min()),
            "max": float(target_data.max()),
            "skewness": float(target_data.skew()),
            "kurtosis": float(target_data.kurtosis())
        }
        
        # Distribution analysis
        hist, bin_edges = np.histogram(target_data, bins='auto')
        analysis["distribution"] = {
            "histogram": {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist()
            }
        }
        
        # Outlier percentage
        q1 = target_data.quantile(0.25)
        q3 = target_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = target_data[(target_data < lower_bound) | (target_data > upper_bound)]
        analysis["outlier_percentage"] = round(len(outliers) / len(target_data) * 100, 2)
        
        # Preprocessing suggestions
        suggestions = []
        if abs(analysis["statistics"]["skewness"]) > 1:
            suggestions.append("Consider log transformation due to high skewness")
        if analysis["outlier_percentage"] > 5:
            suggestions.append(f"Consider outlier handling ({analysis['outlier_percentage']}% outliers detected)")
        if analysis["missing_percentage"] > 0:
            suggestions.append(f"Consider imputation strategy for {analysis['missing_percentage']}% missing values")
        
        analysis["preprocessing_suggestions"] = suggestions
        
    else:
        # Classification target analysis
        value_counts = target_data.value_counts()
        total = len(target_data)
        
        analysis["class_distribution"] = {
            "classes": [str(k) for k in value_counts.index],
            "counts": value_counts.values.tolist(),
            "percentages": [round(v/total*100, 2) for v in value_counts.values]
        }
        
        # Class imbalance analysis
        if len(value_counts) >= 2:
            minority_class_size = value_counts.min()
            majority_class_size = value_counts.max()
            imbalance_ratio = majority_class_size / minority_class_size
            
            analysis["imbalance_analysis"] = {
                "num_classes": int(len(value_counts)),
                "minority_class_size": int(minority_class_size),
                "majority_class_size": int(majority_class_size),
                "imbalance_ratio": round(imbalance_ratio, 2),
                "is_imbalanced": imbalance_ratio > 2
            }
            
            # Preprocessing suggestions
            suggestions = []
            if imbalance_ratio > 2:
                suggestions.append(f"Dataset is imbalanced (ratio: {imbalance_ratio:.2f}). Consider oversampling or class weights")
            if analysis["missing_percentage"] > 0:
                suggestions.append(f"Consider imputation strategy for {analysis['missing_percentage']}% missing values")
            if len(value_counts) > 10:
                suggestions.append(f"High cardinality target ({len(value_counts)} classes). Consider grouping rare classes")
            
            analysis["preprocessing_suggestions"] = suggestions
    
    return analysis

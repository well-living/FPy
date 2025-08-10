# FPy/tests/test_asset_liability_simulator_with_google_sheets.py
# https://github.com/googleapis/google-api-python-client?tab=readme-ov-file

import os
import pytest
import pandas as pd
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# .env読み込み
def load_environment():
    try:
        from dotenv import load_dotenv
        current_file = Path(__file__)
        project_root = current_file.parent.parent if 'tests' in str(current_file) else current_file.parent
        env_path = project_root / '.env'
        
        if env_path.exists():
            load_dotenv(env_path, override=True)
            print(f"Loaded .env from: {env_path}")
            return True
        else:
            print(f".env not found at: {env_path}")
            return False
    except ImportError:
        print("python-dotenv not installed. Run: pip install python-dotenv")
        return False

load_environment()

# 環境変数読み込み後にfpyjpモジュールをインポート
from fpyjp.core.balance_simulator import AssetLiabilitySimulator
from fpyjp.schemas.balance import AssetLiabilitySchema


class GoogleSheetsClient:
    """Google Sheets APIクライアント"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")
        self.service = build('sheets', 'v4', developerKey=api_key)
    
    def get_sheet_data(self, spreadsheet_id: str, sheet_name: str = "Main", range_name: str = None, max_rows: int = None) -> pd.DataFrame:
        """スプレッドシートからデータを取得"""
        try:
            # 範囲の決定ロジック
            if range_name is not None:
                # 明示的に範囲が指定された場合
                full_range = f"{sheet_name}!{range_name}"
            elif max_rows is not None:
                # 最大行数が指定された場合
                full_range = f"{sheet_name}!A1:AB{max_rows}"
            else:
                # どちらも指定されていない場合は全行取得
                full_range = f"{sheet_name}!A:AB"
            
            result = self.service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id, range=full_range
            ).execute()
            
            values = result.get('values', [])
            if not values:
                print("No data found in spreadsheet")
                return pd.DataFrame()
            
            print(f"Retrieved {len(values)} rows from spreadsheet")
            print(f"First row: {values[0][:5]}..." if values[0] else "Empty first row")
            
            # ヘッダー処理（空ヘッダー対応）
            if len(values[0]) == 0 or all(str(cell).strip() == '' for cell in values[0]):
                # ヘッダーが空の場合、自動生成
                max_cols = max(len(row) for row in values) if values else 0
                headers = [f'Column_{i+1}' for i in range(max_cols)]
                data = values
                print(f"Generated headers: {headers[:5]}...")
            else:
                # 通常のヘッダー処理
                headers = [str(cell).strip() for cell in values[0]]
                data = values[1:] if len(values) > 1 else []
                print(f"Found headers: {headers[:5]}...")
            
            # DataFrame作成
            if data:
                max_cols = len(headers)
                normalized_data = [
                    row[:max_cols] + [''] * (max_cols - len(row))
                    for row in data
                ]
                df = pd.DataFrame(normalized_data, columns=headers)
            else:
                df = pd.DataFrame(columns=headers)
            
            print(f"Created DataFrame: {df.shape}")
            
            # 数値列変換
            numeric_columns = [
                'time_period', 'price', 'pre_cash_balance', 'pre_al_unit', 
                'pre_al_balance', 'pre_al_book_balance', 'pre_unrealized_gl',
                'cash_inflow_per_unit', 'income_cash_inflow_before_tax', 
                'income_gain_tax_rate', 'income_gain_tax', 'income_cash_inflow',
                'unit_outflow', 'capital_cash_inflow_before_tax', 
                'capital_gain_tax_rate', 'capital_gain_tax', 'capital_cash_inflow',
                'cash_inflow', 'unit_inflow', 'cash_outflow', 'cash_flow',
                'unit_flow', 'cash_balance', 'al_unit', 'al_balance',
                'al_book_balance', 'unrealized_gl', 'rate'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except HttpError as e:
            if e.resp.status == 403:
                raise Exception("API key invalid or quota exceeded")
            elif e.resp.status == 404:
                raise Exception("Spreadsheet not found or not public")
            else:
                raise Exception(f"API error: {e}")
        except Exception as e:
            print(f"Error in get_sheet_data: {e}")
            raise


def create_simulator_from_spreadsheet(sheet_name: str = "Main", range_name: str = None, max_rows: int = None):
    """スプレッドシートデータからシミュレータを作成"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if not api_key or not sheet_id:
        raise ValueError("Google Sheets environment variables (GOOGLE_SHEETS_API_KEY, TEST_SPREADSHEET_ID) are required")
    
    client = GoogleSheetsClient(api_key)
    df = client.get_sheet_data(sheet_id, sheet_name, range_name, max_rows)
    
    if df.empty:
        raise ValueError("No data retrieved from spreadsheet")
    
    # 必要な列の存在確認
    required_initial_columns = ['price', 'pre_cash_balance', 'pre_al_balance', 'pre_al_book_balance']
    required_series_columns = ['cash_inflow_per_unit', 'income_gain_tax_rate', 'capital_cash_inflow_before_tax', 
                              'capital_gain_tax_rate', 'cash_outflow', 'rate']
    
    missing_initial = [col for col in required_initial_columns if col not in df.columns]
    missing_series = [col for col in required_series_columns if col not in df.columns]
    
    if missing_initial:
        raise ValueError(f"Required initial value columns missing from spreadsheet: {missing_initial}")
    if missing_series:
        raise ValueError(f"Required time series columns missing from spreadsheet: {missing_series}")
    
    # 初期値（最初の行から取得）
    first_row = df.iloc[0]
    initial_price = first_row['price']
    initial_cash_balance = first_row['pre_cash_balance']
    initial_al_balance = first_row['pre_al_balance']
    initial_al_book_balance = first_row['pre_al_book_balance']
    
    # 値の妥当性チェック
    if pd.isna(initial_price):
        raise ValueError("initial_price cannot be NaN")
    if pd.isna(initial_cash_balance):
        raise ValueError("initial_cash_balance cannot be NaN")
    if pd.isna(initial_al_balance):
        raise ValueError("initial_al_balance cannot be NaN")
    if pd.isna(initial_al_book_balance):
        raise ValueError("initial_al_book_balance cannot be NaN")
    
    # 全期間のデータ（リスト）
    cash_inflow_per_unit = df['cash_inflow_per_unit'].tolist()
    income_gain_tax_rate = df['income_gain_tax_rate'].tolist()
    capital_cash_inflow_before_tax = df['capital_cash_inflow_before_tax'].tolist()
    capital_gain_tax_rate = df['capital_gain_tax_rate'].tolist()
    cash_outflow = df['cash_outflow'].tolist()
    rate = df['rate'].tolist()
    
    # 初期単位数を計算
    initial_unit = initial_al_balance / initial_price if initial_price != 0 else 0.0
    
    # AssetLiabilitySchemaを作成
    schema = AssetLiabilitySchema(
        price=initial_price,
        unit=initial_unit,
        balance=initial_al_balance,
        book_balance=initial_al_book_balance,
        cashinflow_per_unit=cash_inflow_per_unit,
        rate=rate
    )
    
    # シミュレータを作成
    simulator = AssetLiabilitySimulator(
        al_schema=schema,
        initial_cash_balance=initial_cash_balance,
        capital_cash_inflow_before_tax=capital_cash_inflow_before_tax,
        cash_outflow=cash_outflow,
        income_gain_tax_rate=income_gain_tax_rate,
        capital_gain_tax_rate=capital_gain_tax_rate
    )
    
    print(f"Created simulator from spreadsheet data: {len(df)} periods")
    return simulator, df


def debug_spreadsheet_data(sheet_name: str = "Main", range_name: str = None, max_rows: int = None):
    """スプレッドシートデータのデバッグ"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if api_key and sheet_id:
        try:
            client = GoogleSheetsClient(api_key)
            df = client.get_sheet_data(sheet_id, sheet_name, range_name, max_rows)
            
            print(f"\n=== Spreadsheet Debug Info ===")
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame empty: {df.empty}")
            print(f"DataFrame columns ({len(df.columns)}): {list(df.columns)}")
            
            if not df.empty:
                print(f"First 3 rows:")
                print(df.head(3))
                print(f"Data types:")
                print(df.dtypes.head(10))
                
                # 初期値の確認
                first_row = df.iloc[0]
                print(f"\nInitial values from spreadsheet:")
                print(f"  price: {first_row.get('price', 'N/A')}")
                print(f"  pre_cash_balance: {first_row.get('pre_cash_balance', 'N/A')}")
                print(f"  pre_al_balance: {first_row.get('pre_al_balance', 'N/A')}")
                print(f"  pre_al_book_balance: {first_row.get('pre_al_book_balance', 'N/A')}")
                
                # 全期間データの確認
                print(f"\nTime series data:")
                for col in ['cash_inflow_per_unit', 'income_gain_tax_rate', 'capital_cash_inflow_before_tax', 
                           'capital_gain_tax_rate', 'cash_outflow', 'rate']:
                    if col in df.columns:
                        values = df[col].tolist()
                        print(f"  {col}: {values}")
            
            print(f"=== Debug End ===\n")
            return df
        except Exception as e:
            print(f"Debug failed: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame()


def test_environment_setup():
    """環境設定テスト"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    assert api_key is not None, "GOOGLE_SHEETS_API_KEY must be set in .env file"
    assert sheet_id is not None, "TEST_SPREADSHEET_ID must be set in .env file"
    assert len(api_key) > 20, f"API key seems too short: {len(api_key)} characters"
    
    print("Environment setup test passed")


def test_api_connection():
    """API接続テスト"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if not api_key or not sheet_id:
        pytest.skip("Environment variables not set")
    
    client = GoogleSheetsClient(api_key)
    df = client.get_sheet_data(sheet_id, "Main", max_rows=5)
    
    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    print(f"API connection test passed: {df.shape}")


def test_spreadsheet_vs_simulation():
    """スプレッドシートとシミュレーション比較テスト"""
    # デバッグ情報を最初に表示
    debug_df = debug_spreadsheet_data()
    
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if not api_key or not sheet_id:
        pytest.skip("Environment variables not set")
    
    # スプレッドシートデータからシミュレータを作成
    simulator, sheet_df = create_simulator_from_spreadsheet()
    
    # シミュレーション実行
    sim_df = simulator.simulate(len(sheet_df))
    
    # 詳細な情報を表示
    print(f"Spreadsheet data: {sheet_df.shape}")
    print(f"Simulation data: {sim_df.shape}")
    print(f"Spreadsheet columns: {list(sheet_df.columns)[:10]}...")
    print(f"Simulation columns: {list(sim_df.columns)[:10]}...")
    
    # 比較対象列
    comparison_columns = ['price', 'cash_balance', 'al_balance', 'unrealized_gl', 'cash_flow']
    tolerance = 1e-2  # 許容誤差を少し大きく
    found_columns = []
    comparison_results = []
    
    for col in comparison_columns:
        if col in sheet_df.columns and col in sim_df.columns:
            found_columns.append(col)
            print(f"Comparing column: {col}")
            
            for i in range(min(len(sim_df), len(sheet_df))):
                sim_val = sim_df.iloc[i][col]
                sheet_val = sheet_df.iloc[i][col]
                
                if pd.notna(sim_val) and pd.notna(sheet_val):
                    diff = abs(sim_val - sheet_val)
                    comparison_results.append({
                        'column': col,
                        'period': i,
                        'sim_val': sim_val,
                        'sheet_val': sheet_val,
                        'diff': diff,
                        'match': diff <= tolerance
                    })
                    
                    if diff > tolerance:
                        print(f"Mismatch {col}[{i}]: sim={sim_val:.6f}, sheet={sheet_val:.6f}, diff={diff:.6f}")
                    else:
                        print(f"Match {col}[{i}]: sim={sim_val:.6f}, sheet={sheet_val:.6f}")
    
    # 比較可能な列が見つからない場合は失敗
    assert len(found_columns) > 0, f"No comparable columns found. Sheet columns: {list(sheet_df.columns)}, Expected: {comparison_columns}"
    
    # 比較結果のサマリー
    total_comparisons = len(comparison_results)
    matches = sum(1 for r in comparison_results if r['match'])
    match_rate = (matches / total_comparisons * 100) if total_comparisons > 0 else 0
    
    print(f"Comparison summary: {matches}/{total_comparisons} matches ({match_rate:.1f}%)")
    
    # 一定の一致率を要求（例：99%以上）
    required_match_rate = 99.0
    assert match_rate >= required_match_rate, f"Match rate {match_rate:.1f}% is below required {required_match_rate}%"
    
    print(f"Spreadsheet vs Simulation comparison passed")


def test_parameter_extraction():
    """スプレッドシートからのパラメータ抽出テスト"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if not api_key or not sheet_id:
        pytest.skip("Environment variables not set")
    
    try:
        client = GoogleSheetsClient(api_key)
        df = client.get_sheet_data(sheet_id, "Main", max_rows=10)
        
        if df.empty:
            pytest.fail("Spreadsheet data is empty")
        
        # 必要な列の存在確認
        required_initial_columns = ['price', 'pre_cash_balance', 'pre_al_balance', 'pre_al_book_balance']
        required_series_columns = ['cash_inflow_per_unit', 'income_gain_tax_rate', 'capital_cash_inflow_before_tax', 
                                  'capital_gain_tax_rate', 'cash_outflow', 'rate']
        
        missing_initial = [col for col in required_initial_columns if col not in df.columns]
        missing_series = [col for col in required_series_columns if col not in df.columns]
        
        # 必要な列が不足している場合はテスト失敗
        assert len(missing_initial) == 0, f"Required initial columns missing: {missing_initial}. Available columns: {list(df.columns)}"
        assert len(missing_series) == 0, f"Required series columns missing: {missing_series}. Available columns: {list(df.columns)}"
        
        # 初期値の妥当性確認
        first_row = df.iloc[0]
        assert not pd.isna(first_row['price']), "price value cannot be NaN"
        assert not pd.isna(first_row['pre_cash_balance']), "pre_cash_balance value cannot be NaN"
        assert not pd.isna(first_row['pre_al_balance']), "pre_al_balance value cannot be NaN"
        assert not pd.isna(first_row['pre_al_book_balance']), "pre_al_book_balance value cannot be NaN"
        
        print("Parameter extraction test passed")
        
    except Exception as e:
        pytest.fail(f"Parameter extraction failed: {e}")





def test_future_value_sheet():
    """終価シートのテスト"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if not api_key or not sheet_id:
        pytest.skip("Environment variables not set")
    
    client = GoogleSheetsClient(api_key)
    df = client.get_sheet_data(sheet_id, "終価", max_rows=20)
    
    assert not df.empty, "終価シートにデータが存在しません"
    
    # 既存の列名を使用して数値変換
    numeric_columns = [
        'time_period', 'price', 'pre_cash_balance', 'pre_al_unit', 
        'pre_al_balance', 'pre_al_book_balance', 'pre_unrealized_gl',
        'cash_inflow_per_unit', 'income_cash_inflow_before_tax', 
        'income_gain_tax_rate', 'income_gain_tax', 'income_cash_inflow',
        'unit_outflow', 'capital_cash_inflow_before_tax', 
        'capital_gain_tax_rate', 'capital_gain_tax', 'capital_cash_inflow',
        'cash_inflow', 'unit_inflow', 'cash_outflow', 'cash_flow',
        'unit_flow', 'cash_balance', 'al_unit', 'al_balance',
        'al_book_balance', 'unrealized_gl', 'rate'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"終価シート列: {list(df.columns)}")
    print(f"終価シートテスト完了: {df.shape}")


def test_present_value_sheet():
    """現価シートのテスト"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if not api_key or not sheet_id:
        pytest.skip("Environment variables not set")
    
    client = GoogleSheetsClient(api_key)
    df = client.get_sheet_data(sheet_id, "現価", max_rows=20)
    
    assert not df.empty, "現価シートにデータが存在しません"
    
    # 既存の列名を使用して数値変換
    numeric_columns = [
        'time_period', 'price', 'pre_cash_balance', 'pre_al_unit', 
        'pre_al_balance', 'pre_al_book_balance', 'pre_unrealized_gl',
        'cash_inflow_per_unit', 'income_cash_inflow_before_tax', 
        'income_gain_tax_rate', 'income_gain_tax', 'income_cash_inflow',
        'unit_outflow', 'capital_cash_inflow_before_tax', 
        'capital_gain_tax_rate', 'capital_gain_tax', 'capital_cash_inflow',
        'cash_inflow', 'unit_inflow', 'cash_outflow', 'cash_flow',
        'unit_flow', 'cash_balance', 'al_unit', 'al_balance',
        'al_book_balance', 'unrealized_gl', 'rate'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"現価シート列: {list(df.columns)}")
    print(f"現価シートテスト完了: {df.shape}")


def test_future_value_annuity_sheet():
    """年金終価シートのテスト"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if not api_key or not sheet_id:
        pytest.skip("Environment variables not set")
    
    client = GoogleSheetsClient(api_key)
    df = client.get_sheet_data(sheet_id, "年金終価", max_rows=20)
    
    assert not df.empty, "年金終価シートにデータが存在しません"
    
    # 既存の列名を使用して数値変換
    numeric_columns = [
        'time_period', 'price', 'pre_cash_balance', 'pre_al_unit', 
        'pre_al_balance', 'pre_al_book_balance', 'pre_unrealized_gl',
        'cash_inflow_per_unit', 'income_cash_inflow_before_tax', 
        'income_gain_tax_rate', 'income_gain_tax', 'income_cash_inflow',
        'unit_outflow', 'capital_cash_inflow_before_tax', 
        'capital_gain_tax_rate', 'capital_gain_tax', 'capital_cash_inflow',
        'cash_inflow', 'unit_inflow', 'cash_outflow', 'cash_flow',
        'unit_flow', 'cash_balance', 'al_unit', 'al_balance',
        'al_book_balance', 'unrealized_gl', 'rate'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"年金終価シート列: {list(df.columns)}")
    print(f"年金終価シートテスト完了: {df.shape}")


def test_sinking_fund_sheet():
    """減債基金係数シートのテスト"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if not api_key or not sheet_id:
        pytest.skip("Environment variables not set")
    
    client = GoogleSheetsClient(api_key)
    df = client.get_sheet_data(sheet_id, "減債基金係数", max_rows=20)
    
    assert not df.empty, "減債基金係数シートにデータが存在しません"
    
    # 既存の列名を使用して数値変換
    numeric_columns = [
        'time_period', 'price', 'pre_cash_balance', 'pre_al_unit', 
        'pre_al_balance', 'pre_al_book_balance', 'pre_unrealized_gl',
        'cash_inflow_per_unit', 'income_cash_inflow_before_tax', 
        'income_gain_tax_rate', 'income_gain_tax', 'income_cash_inflow',
        'unit_outflow', 'capital_cash_inflow_before_tax', 
        'capital_gain_tax_rate', 'capital_gain_tax', 'capital_cash_inflow',
        'cash_inflow', 'unit_inflow', 'cash_outflow', 'cash_flow',
        'unit_flow', 'cash_balance', 'al_unit', 'al_balance',
        'al_book_balance', 'unrealized_gl', 'rate'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"減債基金係数シート列: {list(df.columns)}")
    print(f"減債基金係数シートテスト完了: {df.shape}")


def test_capital_recovery_sheet():
    """資本回収係数シートのテスト"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if not api_key or not sheet_id:
        pytest.skip("Environment variables not set")
    
    client = GoogleSheetsClient(api_key)
    df = client.get_sheet_data(sheet_id, "資本回収係数", max_rows=20)
    
    assert not df.empty, "資本回収係数シートにデータが存在しません"
    
    # 既存の列名を使用して数値変換
    numeric_columns = [
        'time_period', 'price', 'pre_cash_balance', 'pre_al_unit', 
        'pre_al_balance', 'pre_al_book_balance', 'pre_unrealized_gl',
        'cash_inflow_per_unit', 'income_cash_inflow_before_tax', 
        'income_gain_tax_rate', 'income_gain_tax', 'income_cash_inflow',
        'unit_outflow', 'capital_cash_inflow_before_tax', 
        'capital_gain_tax_rate', 'capital_gain_tax', 'capital_cash_inflow',
        'cash_inflow', 'unit_inflow', 'cash_outflow', 'cash_flow',
        'unit_flow', 'cash_balance', 'al_unit', 'al_balance',
        'al_book_balance', 'unrealized_gl', 'rate'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"資本回収係数シート列: {list(df.columns)}")
    print(f"資本回収係数シートテスト完了: {df.shape}")


def test_present_value_annuity_sheet():
    """年金現価シートのテスト"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if not api_key or not sheet_id:
        pytest.skip("Environment variables not set")
    
    client = GoogleSheetsClient(api_key)
    df = client.get_sheet_data(sheet_id, "年金現価", max_rows=20)
    
    assert not df.empty, "年金現価シートにデータが存在しません"
    
    # 既存の列名を使用して数値変換
    numeric_columns = [
        'time_period', 'price', 'pre_cash_balance', 'pre_al_unit', 
        'pre_al_balance', 'pre_al_book_balance', 'pre_unrealized_gl',
        'cash_inflow_per_unit', 'income_cash_inflow_before_tax', 
        'income_gain_tax_rate', 'income_gain_tax', 'income_cash_inflow',
        'unit_outflow', 'capital_cash_inflow_before_tax', 
        'capital_gain_tax_rate', 'capital_gain_tax', 'capital_cash_inflow',
        'cash_inflow', 'unit_inflow', 'cash_outflow', 'cash_flow',
        'unit_flow', 'cash_balance', 'al_unit', 'al_balance',
        'al_book_balance', 'unrealized_gl', 'rate'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"年金現価シート列: {list(df.columns)}")
    print(f"年金現価シートテスト完了: {df.shape}")


def test_coupon_bond_sheet():
    """利付債シートのテスト"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if not api_key or not sheet_id:
        pytest.skip("Environment variables not set")
    
    client = GoogleSheetsClient(api_key)
    df = client.get_sheet_data(sheet_id, "利付債", max_rows=20)
    
    assert not df.empty, "利付債シートにデータが存在しません"
    
    # 既存の列名を使用して数値変換
    numeric_columns = [
        'time_period', 'price', 'pre_cash_balance', 'pre_al_unit', 
        'pre_al_balance', 'pre_al_book_balance', 'pre_unrealized_gl',
        'cash_inflow_per_unit', 'income_cash_inflow_before_tax', 
        'income_gain_tax_rate', 'income_gain_tax', 'income_cash_inflow',
        'unit_outflow', 'capital_cash_inflow_before_tax', 
        'capital_gain_tax_rate', 'capital_gain_tax', 'capital_cash_inflow',
        'cash_inflow', 'unit_inflow', 'cash_outflow', 'cash_flow',
        'unit_flow', 'cash_balance', 'al_unit', 'al_balance',
        'al_book_balance', 'unrealized_gl', 'rate'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"利付債シート列: {list(df.columns)}")
    print(f"利付債シートテスト完了: {df.shape}")


def test_equal_principal_repayment_sheet():
    """元金均等返済シートのテスト"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if not api_key or not sheet_id:
        pytest.skip("Environment variables not set")
    
    client = GoogleSheetsClient(api_key)
    df = client.get_sheet_data(sheet_id, "元金均等返済", max_rows=50)
    
    assert not df.empty, "元金均等返済シートにデータが存在しません"
    
    # 既存の列名を使用して数値変換
    numeric_columns = [
        'time_period', 'price', 'pre_cash_balance', 'pre_al_unit', 
        'pre_al_balance', 'pre_al_book_balance', 'pre_unrealized_gl',
        'cash_inflow_per_unit', 'income_cash_inflow_before_tax', 
        'income_gain_tax_rate', 'income_gain_tax', 'income_cash_inflow',
        'unit_outflow', 'capital_cash_inflow_before_tax', 
        'capital_gain_tax_rate', 'capital_gain_tax', 'capital_cash_inflow',
        'cash_inflow', 'unit_inflow', 'cash_outflow', 'cash_flow',
        'unit_flow', 'cash_balance', 'al_unit', 'al_balance',
        'al_book_balance', 'unrealized_gl', 'rate'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"元金均等返済シート列: {list(df.columns)}")
    print(f"元金均等返済シートテスト完了: {df.shape}")


def test_equal_payment_repayment_sheet():
    """元利均等返済シートのテスト"""
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    sheet_id = os.getenv('TEST_SPREADSHEET_ID')
    
    if not api_key or not sheet_id:
        pytest.skip("Environment variables not set")
    
    client = GoogleSheetsClient(api_key)
    df = client.get_sheet_data(sheet_id, "元利均等返済", max_rows=50)
    
    assert not df.empty, "元利均等返済シートにデータが存在しません"
    
    # 既存の列名を使用して数値変換
    numeric_columns = [
        'time_period', 'price', 'pre_cash_balance', 'pre_al_unit', 
        'pre_al_balance', 'pre_al_book_balance', 'pre_unrealized_gl',
        'cash_inflow_per_unit', 'income_cash_inflow_before_tax', 
        'income_gain_tax_rate', 'income_gain_tax', 'income_cash_inflow',
        'unit_outflow', 'capital_cash_inflow_before_tax', 
        'capital_gain_tax_rate', 'capital_gain_tax', 'capital_cash_inflow',
        'cash_inflow', 'unit_inflow', 'cash_outflow', 'cash_flow',
        'unit_flow', 'cash_balance', 'al_unit', 'al_balance',
        'al_book_balance', 'unrealized_gl', 'rate'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"元利均等返済シート列: {list(df.columns)}")
    print(f"元利均等返済シートテスト完了: {df.shape}")



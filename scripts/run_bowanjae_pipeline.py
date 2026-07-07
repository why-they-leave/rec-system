import os
import sys
import logging
import pandas as pd

# 프로젝트 루트 경로를 시스템 패스에 추가 (모듈 로딩용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 바뀐 파이프라인 모듈/함수 가져오기
from src.modeling.complementary.model import run_modeling
from src.evaluation.evaluate_complementary import evaluate_model

def setup_logging():
    """로깅 전역 설정 및 파일 저장 설정"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "bowanjae_pipeline.log"), encoding="utf-8")
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Complementary (Bowanjae) Recommendation Pipeline.")

    # 1. 경로 설정 (프로젝트 루트 기준 상대 경로 설정)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_dir = os.path.join(base_dir, "data", "processed")
    raw_data_dir = os.path.join(base_dir, "data", "raw")
    
    # 출력 경로 설정 (backend/api/services/complementary_service.py가 읽는 경로)
    output_dir = os.path.join(base_dir, "data", "outputs", "complementary")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 2. 전처리 완료 로그 및 상품 메타데이터 로드
        logger.info("Step 1: Loading preprocessed logs and product metadata.")
        df_integrated_logs = pd.read_csv(os.path.join(processed_data_dir, "df_integrated_logs.csv"))
        products = pd.read_csv(os.path.join(raw_data_dir, "products.csv"))
        
        # 3. 모델링 파이프라인 실행 (조건부 확률 계산 및 Top 5 추출)
        top_n_recs, df_recs, train_df, test_df = run_modeling(df_integrated_logs, products)
        
        # 4. 평가 파이프라인 실행 (가설 검증 및 Hit Rate 산출)
        evaluate_model(train_df, test_df, df_recs)
        
        # 5. 데이터 인터페이스 통일 및 파일 저장 (Issue #4 대응)
        logger.info("Transforming output schema to match unified data interface.")
        result = (
            top_n_recs
            .rename(columns={"prod_A": "item_id", "prod_B": "rec_item_id"})
            [["item_id", "rec_item_id", "score", "rank"]]
        )
        
        output_path = os.path.join(output_dir, "detail_cf.csv")
        result.to_csv(output_path, index=False)
        logger.info(f"Pipeline successfully finished. Output saved to {output_path}")

    except Exception as e:
        logger.exception(f"Pipeline failed due to an error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
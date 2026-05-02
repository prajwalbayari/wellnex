import asyncio, sys, json
from pathlib import Path
sys.path.insert(0, r'c:\Users\Prajwal\Desktop\Wellnex\WellnexWeb\ml-service')
import main

async def run():
    for d in ['diabetes','heart','breast']:
        main._load_disease(d)
    print('Loaded models:', list(main.models.keys()))

    # Diabetes high-risk
    di = {
        'age':55,'gender':'Male','bmi':32.0,'systolic_bp':150.0,'diastolic_bp':95.0,
        'glucose_fasting':160.0,'glucose_postprandial':250.0,'insulin_level':25.0,'hba1c':9.1,
        'cholesterol_total':260.0,'hdl_cholesterol':35.0,'ldl_cholesterol':160.0,'triglycerides':220.0
    }
    try:
        res = await main.predict_diabetes(main.DiabetesInput(**di))
        print('Diabetes high-risk ->', res.dict())
    except Exception as e:
        print('Diabetes high-risk error:', e)

    # Diabetes low-risk
    di2 = dict(di)
    di2.update({'age':30,'bmi':22.0,'systolic_bp':118.0,'diastolic_bp':76.0,'glucose_fasting':90.0,'glucose_postprandial':110.0,'insulin_level':8.0,'hba1c':5.4,'cholesterol_total':170,'hdl_cholesterol':55,'ldl_cholesterol':90,'triglycerides':120})
    try:
        res = await main.predict_diabetes(main.DiabetesInput(**di2))
        print('Diabetes low-risk ->', res.dict())
    except Exception as e:
        print('Diabetes low-risk error:', e)

    # Heart high-risk
    hi = {'male':1.0,'age':68,'currentSmoker':1.0,'totChol':280,'sysBP':165,'diaBP':100,'BMI':32.0,'heartRate':95,'glucose':150}
    try:
        res = await main.predict_heart(main.HeartInput(**hi))
        print('Heart high-risk ->', res.dict())
    except Exception as e:
        print('Heart high-risk error:', e)

    # Heart low-risk
    hi2 = {'male':0.0,'age':35,'currentSmoker':0.0,'totChol':170,'sysBP':118,'diaBP':76,'BMI':22.0,'heartRate':72,'glucose':92}
    try:
        res = await main.predict_heart(main.HeartInput(**hi2))
        print('Heart low-risk ->', res.dict())
    except Exception as e:
        print('Heart low-risk error:', e)

    # Unified PDF tests
    base = Path(r'c:\Users\Prajwal\Desktop\Wellnex\WellnexWeb\reports\test_pdfs')
    for fname in ['heart_positive_1.pdf','heart_negative_1.pdf','diabetes_positive_1.pdf']:
        p = base / fname
        if p.exists():
            class DummyFile:
                def __init__(self,path):
                    self.filename = Path(path).name
                    self.content_type = 'application/pdf'
                    self._bytes = Path(path).read_bytes()
                async def read(self):
                    return self._bytes
            try:
                out = await main.predict_unified(DummyFile(str(p)), supplemental_data='{}')
                print(p.name, '-> summary:', json.dumps(out['summary']))
                print(p.name, '-> model_results keys:', list(out['model_results'].keys()))
            except Exception as e:
                print(p.name, 'error:', e)
        else:
            print('Missing test PDF:', p)

if __name__ == '__main__':
    asyncio.run(run())

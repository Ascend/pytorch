import time
import unittest
import os

def load_local_case(test_case_path):
    discover=unittest.defaultTestLoader.discover(test_case_path, "test_*.py")
    return discover

def run_tests():

    test_case_path='./'
    test_report_path=test_case_path+'ReportResult'

    ENABLE_HTML=bool(os.environ.get('ENABLE_HTML'))
    ENABLE_HTML_MX=bool(os.environ.get('ENABLE_HTML_MX'))
    ENABLE_CASE_PATH=os.environ.get('ENABLE_CASE_PATH')
    ENABLE_OUTPUT_PATH=os.environ.get('ENABLE_OUTPUT_PATH')

    if ENABLE_CASE_PATH is not None:
        if not os.path.exists(ENABLE_CASE_PATH):
            print('path is not exists: ', ENABLE_CASE_PATH)
        else:
            test_case_path=ENABLE_CASE_PATH
            test_report_path=test_case_path+'ReportResult'

    if ENABLE_OUTPUT_PATH is not None:
        if not os.path.exists(ENABLE_OUTPUT_PATH):
            print('path is not exists: ', ENABLE_OUTPUT_PATH)
        else:
            test_report_path=ENABLE_OUTPUT_PATH

    if not os.path.exists(test_report_path):
        os.mkdir(test_report_path)
        print(test_report_path)

    now=time.strftime("%Y_%m_%d_%H_%M_%S")
    htmlFileName=os.path.join(test_report_path, 'pytorch-unittest-report-'+now+'.html')
    txtFileName=os.path.join(test_report_path, 'pytorch-unittest-report-'+now+'.txt')

    if ENABLE_HTML:
        print('start pytorch HTML unittest testset...')
        import HTMLTestRunner
        with open(htmlFileName, "wb") as report_file:
            runner=HTMLTestRunner.HTMLTestRunner(stream=report_file, title='AllTest', description='all npu test case', verbosity=2)
            runner.run(load_local_case(test_case_path))
        print('report files path', htmlFileName)
    elif ENABLE_HTML_MX:
        print('start pytorch Multi HTML unittest testset...')
        import HtmlTestRunner
        runner=HtmlTestRunner.HTMLTESTRunner(output=test_report_path, verbosity=2)
        runner=run(load_local_case(test_case_path))
    else:
        print('start pytorch TEXT unittest testset...')
        with open(txtFileName, "a") as report_file:
            runner=unittest.TextTestRunner(stream=report_file, verbosity=2)
            runner.run(load_local_case(test_case_path))
        print('report files path', txtFileName)

if __name__=="__main__":
    run_tests()



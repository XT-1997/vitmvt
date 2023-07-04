from mmcv.utils import import_modules_from_strings


def single_gpu_test(model, data_loader, test_setting):
    scope = test_setting.repo
    pac_name = scope + '.apis'
    test_func = import_modules_from_strings(pac_name).single_gpu_test
    return test_func(model, data_loader, **test_setting.single_gpu_test)


def multi_gpu_test(model, data_loader, test_setting):
    scope = test_setting.repo
    pac_name = scope + '.apis'
    test_func = import_modules_from_strings(pac_name).multi_gpu_test
    return test_func(model, data_loader, **test_setting.multi_gpu_test)

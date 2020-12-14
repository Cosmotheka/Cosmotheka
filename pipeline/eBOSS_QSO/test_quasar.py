from QuasarMapper import QuasarMapper

data_path = '/home/zcapjru/PhD/Data/'
input_1 = [data_path+'eBOSS_QSO_clustering_data-NGC-vDR16.fits', data_path+'eBOSS_QSO_clustering_data-SGC-vDR16.fits']
input_2 = [data_path+'eBOSS_QSO_clustering_random-NGC-vDR16.fits', data_path+'eBOSS_QSO_clustering_random-SGC-vDR16.fits']

test_example = QuasarMapper(input_1, input_2)

print(test_example.get_nz(1))
print(test_example.get_delta_map(1))
print(test_example.get_nmean_map(1))
print(test_example.get_mask(1))
print(test_example.get_nl(1,1))

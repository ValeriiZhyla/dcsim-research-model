DIR="D:\\KIT\\Master Thesis\\simulations\\fourth_phase"

PREFIX_1="1-sgbatch-normal-simple-jobs"
PREFIX_2="2-sgbatch-high-links-latency-simple-jobs"
PREFIX_3="3-sgbatch-low-link-bandwidth-medium-latency-complex-jobs"
PREFIX_4="4-sgbatch-more-cores-complex-jobs"
PREFIX_5="5-sgbatch-less-cores-more-ram"
PREFIX_6="6-sgbatch-slower-storage-disk"
PREFIX_7="7-sgbatch-one-host-less"
PREFIX_8="8-kit_desy-normal"
PREFIX_9="9-kit_desy-no-fatpipes"
PREFIX_10="10-kit_desy-high-link-bandwidth"
PREFIX_11="11-kit_desy-less-cores-everywhere"
PREFIX_12="12-kit_desy-cache-moved-to-desy"
PREFIX_13="13-kit_desy-cache-moved-to-desy-faster-cores-mf-everywhere"
PREFIX_14="14-kit_desy-cache-moved-to-desy-less-cores-and-ram-in-kit"
PREFIX_15="15-kit_desy-cache-moved-to-desy-more-cores-in-desy"

prefixes = [PREFIX_1, PREFIX_2, PREFIX_3, PREFIX_4, PREFIX_5, PREFIX_6, PREFIX_7, PREFIX_8, PREFIX_9, PREFIX_10, PREFIX_11, PREFIX_12, PREFIX_13, PREFIX_14, PREFIX_15]
paths = map(lambda item: f"{DIR}\\{item}", prefixes)

for path in paths:
    print(f'python extract-simulations.py --simulation_root_dir="{path}\\5"')
    print(f'python extract-simulations.py --simulation_root_dir="{path}\\10"')
    print(f'python extract-simulations.py --simulation_root_dir="{path}\\20"')
    print(f'python extract-simulations.py --simulation_root_dir="{path}\\50"')
    print(f'python extract-simulations.py --simulation_root_dir="{path}\\100"')
    print(f'python extract-simulations.py --simulation_root_dir="{path}\\250"')
    print(f'python extract-simulations.py --simulation_root_dir="{path}\\500"')
    print(f'python extract-simulations.py --simulation_root_dir="{path}\\1000"')
    print(f'python extract-simulations.py --simulation_root_dir="{path}\\1500"')
    print(f'python extract-simulations.py --simulation_root_dir="{path}\\2000"')
    print()


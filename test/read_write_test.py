from lasso.dyna import D3plot

filelist = [
    "/home/codie/programming/python/d3plot-test-data/jmaestre2/d3plot",
    "/home/codie/programming/python/d3plot-test-data/mattb5/d3plot.fz",
    "/home/codie/programming/python/d3plot-test-data/walzen/stich/d3plot",
    "/home/codie/programming/python/d3plot-test-data/walzen/transport/d3plot",
    "/home/codie/programming/python/d3plot-test-data/femzipped/d3plot1.fz",
    "/home/codie/programming/python/d3plot-test-data/devendra/d3plot",
    "/home/codie/programming/python/d3plot-test-data/kracker1/022_021_Validierung_anfang_fehlerfreig",
    "/home/codie/programming/python/d3plot-test-data/kracker1/r_022_021_Validierung_anfang_fehlerfreig",
    "/home/codie/programming/python/d3plot-test-data/kracker1/z_r_022_021_Validierung_anfang_fehlerfreig",
    "/home/codie/programming/python/d3plot-test-data/plotcompress/cpr_d3plot",
    "/home/codie/programming/python/d3plot-test-data/mattb1/LASSO_P1_0_DROP_-_FIX_BR192407.d3plot",
    "/home/codie/programming/python/d3plot-test-data/mattb3/d3plot.fz",
    "/home/codie/programming/python/d3plot-test-data/fau3/COFDC3_P1_2A_FC_THOR50_8WP_AVS1-1G6B2_Relax_GR200249.d3plot",
    "/home/codie/programming/python/d3plot-test-data/mattb4/LASSO_P1_0_DROP_-_FIX_BR200993.d3plot",
    "/home/codie/programming/python/d3plot-test-data/mattb4/LASSO_P1_0_DROP_-_FIX_BR200993_modif.d3plot",
]

def main():
    for filepath in filelist:
        print(60*"=")
        print(filepath)
        
        d3plot = D3plot(filepath)
        d3plot.write_d3plot("/tmp/d3plot_test")
        d3plot2 = D3plot("/tmp/d3plot_test")
        
        header_diff, arr_diff = d3plot.compare(d3plot2)
        
        print("---------- HEADER DIFF -----------")
        for name, msg in header_diff.items():
            print(name, msg)
        print("---------- ARRAY DIFF -----------")
        for name, msg in arr_diff.items():
            print(name, msg)

if __name__ == "__main__":
    main()

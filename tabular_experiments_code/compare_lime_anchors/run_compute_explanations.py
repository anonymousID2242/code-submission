import subprocess
datasets = ['adult', 'germancredit', 'abalone']
explainers = ['anchor', 'lime']
models = ['nn']
out = 'out_pickles'
for dataset in datasets:
    for explainer in explainers:
        for model in models:
            outfile = '/tmp/%s-%s-%s.log' % (dataset, explainer, model)
            print 'Outfile:', outfile
            outfile = open(outfile, 'w', 0)
            cmd = 'python compute_explanations.py -d %s -e %s -m %s -o %s' % (
                dataset, explainer, model,
                '%s/%s-%s-%s' % (out, dataset, explainer, model))
            print cmd
            subprocess.Popen(cmd.split(), stdout=outfile)

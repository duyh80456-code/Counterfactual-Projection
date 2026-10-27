"""Independent executable H2 feature-geometry learner, sharing native CIL I/O."""
from experiments.standard_cl import parser,run


if __name__=='__main__':
    p=parser();p.set_defaults(branch='H2',arm='h2',config='configs/standard_cl/h2_cifar100_bn.json')
    run(p.parse_args())

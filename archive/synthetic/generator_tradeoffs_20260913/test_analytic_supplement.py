"""Post-run independent validation of newly added analytic references."""
import unittest
import numpy as np
import analytic_effects as a
import benchmark as b

class AnalyticSupplementTests(unittest.TestCase):
    def test_student_gaussian_limit(self):
        mean=np.array([.2,-.3,0.]);cov=b.L@b.L.T
        for center in [-.4,.2,1.]:
            actual=a.student_arm(mean,cov,100000,center,.17)[0]
            expected=a.gaussian_arm(mean,cov,center,.17)[0]
            self.assertAlmostEqual(actual,expected,places=4)

    def test_teacher_against_independent_direct_samples(self):
        # Central, well-supported queries: six independent banks estimate
        # the Monte Carlo uncertainty of the comparison, not just its value.
        for law in b.LAWS:
            h=np.array([.1,-.2,.3]);centers=[-.2,.4];bw=.3
            exact=a.teacher(h,law,centers,bw)
            estimates=[]
            for seed in range(6):
                y=b.oracle(h,200000,87000+seed,law)[0]
                means=[]
                for center in centers:
                    w=np.exp(-.5*((y[:,0]-center)/bw)**2)
                    means.append(np.average(y[:,1],weights=w))
                estimates.append(means[1]-means[0])
            se=np.std(estimates,ddof=1)/np.sqrt(len(estimates))
            self.assertLess(abs(np.mean(estimates)-exact),max(6*se,.001))

    def test_student_symmetry_and_zero_cross_covariance(self):
        mean=np.array([.2,-.3,0.]);cov=b.L@b.L.T
        self.assertAlmostEqual(a.student_arm(mean,cov,4,.2,.1)[0],-.3,places=10)
        independent=np.diag(np.diag(cov))
        self.assertAlmostEqual(a.student_arm(mean,independent,4,1.,.1)[0],-.3,places=10)

if __name__=='__main__':unittest.main(verbosity=2)

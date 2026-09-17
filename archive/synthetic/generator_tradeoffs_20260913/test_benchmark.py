import unittest
import numpy as np
import torch
import benchmark as b

class Checks(unittest.TestCase):
    def test_true_derivative(self):
        for law in b.LAWS:
            for h in b.CENTERS.values():
                d=np.array([1e-5,0,0])
                numerical=(b.true_mean(h+d,law)-b.true_mean(h-d,law))[0]/2e-5
                np.testing.assert_allclose(numerical,b.true_derivative(h,law),atol=1e-7)
    def test_gaussian_soft_conditioning_oracle(self):
        h=np.array([.2,-.3,.1]);centers=np.array([-.3,.4]);bw=.2
        covariance=b.L@b.L.T
        expected=covariance[1,0]/(covariance[0,0]+bw**2)*(centers[1]-centers[0])
        estimate,ess,_=b.weighted(b.oracle(h,300000,121,'linear_gaussian')[0],centers,bw)
        self.assertGreater(ess,10000);self.assertLess(abs(estimate-expected),.003)
    def test_oracle_means(self):
        for law in b.LAWS:
            h=np.array([.5,-.2,.1]);draw=b.oracle(h,200000,971,law)[0]
            np.testing.assert_allclose(draw.mean(0),b.true_mean(h,law)[0],atol=.005)
    def test_energy_deterministic_limit(self):
        x=np.zeros((4,3));y=np.ones((8,3));self.assertAlmostEqual(b.energy(x,y),np.sqrt(3))
    def test_neural_interfaces(self):
        h=torch.randn(8,3);y=torch.randn(8,3)
        for family in b.FAMILIES[2:]:
            m=b.neural(family);loss=m.native_loss(h,y);self.assertTrue(torch.isfinite(loss));loss.backward();m.eval()
            with torch.no_grad():a=m.sample(h,8,212);c=m.sample(h,8,212)
            self.assertEqual(tuple(a.shape),(8,8,3));self.assertTrue(torch.isfinite(a).all());torch.testing.assert_close(a,c)
    def test_event_pairing_zero(self):
        draw=b.oracle([0,0,0],1000,92,'bimodal')[0]
        self.assertEqual(b.weighted(draw,[.2,.2],.1)[0],0.)

if __name__=='__main__':unittest.main()

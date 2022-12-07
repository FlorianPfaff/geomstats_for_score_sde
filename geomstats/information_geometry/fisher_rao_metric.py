"""Class to implement simply the Fisher-Rao metric on information manifolds."""

from scipy.integrate import quad_vec

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric


class FisherRaoMetric(RiemannianMetric):
    r"""Class to derive the information metric from the pdf in InformationManifoldMixin.

    Given a statistical manifold with coordinates :math:`\theta`,
    the Fisher information metric is:
    :math:`g_{j k}(\theta)=\int_X \frac{\partial \log p(x, \theta)}{\partial \theta_j}
        \frac{\partial \log p(x, \theta)}{\partial \theta_k} p(x, \theta) d x`

    Attributes
    ----------
    information_manifold : InformationManifoldMixin object
        Riemannian Manifold for a parametric family of (real) distributions.
    support : list, shape = (2,)
        Left and right bounds for the support of the distribution.
        But this is just to help integration, bounds should be as large as needed.
    """

    def __init__(self, information_manifold, support):
        super().__init__(
            dim=information_manifold.dim,
            shape=(information_manifold.dim,),
            signature=(information_manifold.dim, 0),
        )
        self.information_manifold = information_manifold
        self.support = support

    def metric_matrix(self, base_point):
        r"""Compute the inner-product matrix.

        The Fisher information matrix is noted I in the literature.

        Compute the inner-product matrix of the Fisher information metric
        at the tangent space at base point.

        .. math::

            I(\theta) = \mathbb{E}[YY^T]

        where,

        ..math::

            Y = \nabla_{\theta} \log f_{\theta}(x)


        After manipulation and in indicial notation

        .. math::
             I_{ij} = \int \
             \partial_{i} f_{\theta}(x)\
             \partial_{j} f_{\theta}(x)\
             \frac{1}{f_{\theta}(x)}


        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.

        References
        ----------
        .. [AS1985] Amari, S (1985)
            Differential Geometric Methods in Statistics, Berlin, Springer – Verlag.
        """

        def log_pdf_at_x(x):
            """Compute the log function of the pdf for a fixed value on the support.

            Parameters
            ----------
            x : float, shape (,)
                Point on the support of the distribution
            """
            return lambda point: gs.log(
                self.information_manifold.point_to_pdf(point)(x)
            )

        def log_pdf_derivative(x):
            """Compute the derivative of the log-pdf with respect to the parameters.

            Parameters
            ----------
            x : float, shape (,)
                Point on the support of the distribution
            """
            return gs.autodiff.jacobian(log_pdf_at_x(x))(base_point)

        def log_pdf_derivative_squared(x):
            """Compute the square (in matrix terms) of dlog.

            This is the variable whose expectance is the Fisher-Rao information.

            Parameters
            ----------
            x : float, shape (,)
                Point on the support of the distribution
            """
            dlog = log_pdf_derivative(x)
            return gs.einsum("...i, ...j -> ...ij", dlog, dlog)

        metric_mat = quad_vec(
            lambda x: log_pdf_derivative_squared(x)
            * self.information_manifold.point_to_pdf(base_point)(x),
            *self.support
        )[0]

        if metric_mat.ndim == 3 and metric_mat.shape[0] == 1:
            return metric_mat[0]
        return metric_mat

    def inner_product_derivative_matrix(self, base_point):
        r"""Compute the derivative of the inner-product matrix.

        Compute the derivative of the inner-product matrix of
        the Fisher information metric at the tangent space at base point.

        .. math::

            I(\theta) = \partial_{\theta} \mathbb{E}[YY^T]

        where,

        ..math::

            Y = \nabla_{\theta} \log f_{\theta}(x)

        or, in indicial notation:

        .. math::

            \partial_k I_{ij} = \int\
            \partial_{ki}^2 f\partial_j f \frac{1}{f} + \
            \partial_{kj}^2 f\partial_i f \frac{1}{f} - \
            \partial_i f \partial_j f \partial_k f \frac{1}{f^2}

        with :math:`f = f_{\theta}(x)`


        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Derivative of the inner-product matrix.
        """

        def pdf(x):
            """Compute pdf at a fixed point on the support.

            Parameters
            ----------
            x : float, shape (,)
                Point on the support of the distribution
            """
            return lambda point: self.information_manifold.point_to_pdf(point)(x)

        def _function_to_integrate(x):
            pdf_x = pdf(x)
            pdf_x_at_base_point = pdf_x(base_point)
            pdf_x_derivative = gs.autodiff.jacobian(pdf_x)
            pdf_x_derivative_at_base_point = pdf_x_derivative(base_point)
            pdf_x_hessian_at_base_point = gs.autodiff.jacobian(pdf_x_derivative)(
                base_point
            )
            return (
                1
                / (pdf_x_at_base_point**2)
                * (
                    pdf_x_at_base_point
                    * (
                        gs.einsum(
                            "...ki,...j->...ijk",
                            pdf_x_hessian_at_base_point,
                            pdf_x_derivative_at_base_point,
                        )
                        + gs.einsum(
                            "...kj,...i->...ijk",
                            pdf_x_hessian_at_base_point,
                            pdf_x_derivative_at_base_point,
                        )
                    )
                    - gs.einsum(
                        "...i, ...j, ...k -> ...ijk",
                        pdf_x_derivative_at_base_point,
                        pdf_x_derivative_at_base_point,
                        pdf_x_derivative_at_base_point,
                    )
                )
            )

        return quad_vec(_function_to_integrate, *self.support)[0]

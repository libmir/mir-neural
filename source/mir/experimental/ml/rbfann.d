/++
Radial basis function neural network.
+/
module mir.experimental.ml.rbfann;

import mir.ndslice.slice;
import mir.ndslice.selection : reshape;

import mir.experimental.model.rbf;

/++
Radial basis function artificial neural network model.

Single-layered, linear, non-parametric artificial neural network model.

Params:
    T = Value type used. Default is double.
    rbf = Radial basis function used. Default is gaussian.

See:
    mir.experimental.model.rbf
+/
struct RbfNetwork(T = double, alias rbf = gaussian!T)
{
    static assert (isFloatingPoint!T, "RbfNetwork value type has to be a floating point.");

private:

    alias Vector = Slice!(1, T*);
    alias Matrix = Slice!(2, T*);

    Matrix _centers; // training centers (input layer of the network).
    Matrix _weights; // calculated training weights (hidden layer of the network).

    T _radius; // radius of a basis function.
    T _lambda; // regularization value for ridge regression.

public:

    @disable this();

    this(T radius = 1.0, T lambda = 1e-4)
    {
        this._radius = radius;
        this._lambda = lambda;
    }

    ref radius(in T value) @property
    {
        this._radius = value;
        return this;
    }

    ref lambda(in T value) @property
    {
        this._lambda = value;
        return this;
    }

    auto radius() const @property
    {
        return this._radius;
    }

    auto lambda() const @property
    {
        return this._lambda;
    }

    auto isTrained() const @property
    {
        return !_centers.empty;
    }

    /++
     Train neural network with given data.

     Params:
         x = Center values, data node positions.
         y = Training value, outcome of trained function at x center position.
     Returns:
         Trained model.
    +/
    ref train(Vector x, Vector y)
    {
        return train(x.reshape(x.length, 1), y.reshape(y.length, 1));
    }

    /// ditto
    ref train(Matrix x, Matrix y)
    in
    {
        assert(x.length == y.length);
    }
    body
    {
        this._centers = x.slice;
        this._weights = slice!T(y.shape);

        auto H = slice!T(x.length, x.length);

        designRbf!rbf(x, x, _radius, H);
        rigdeGlobalWeights(y, H, _lambda, _weights);

        return this;
    }

    /++
     Perform data fitting based on previous neural network training.
    +/
    Vector fit(Vector qx)
    {
        auto ft = fit(qx.reshape(qx.length, 1));
        return ft.reshape(qx.length);
    }

    /// ditto
    Matrix fit(Matrix qx)
    in
    {
        assert(qx.length!1 == _centers.length!1);
    }
    body
    {
        auto pq = qx.length;
        auto p = this._weights.length!0;
        auto d = this._weights.length!1;
        auto Ht = slice!T(pq, p);
        auto ft = slice!T(pq, d);

        designRbf!rbf(qx, this._centers, _radius, Ht);

        gemm(1.0, Ht, this._weights, 0.0, ft);

        return ft;
    }
}

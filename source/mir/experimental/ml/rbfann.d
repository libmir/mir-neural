/++
Radial basis function neural network.
+/
module mir.experimental.ml.rbfann;

import mir.ndslice.slice;
import mir.ndslice.selection : reshape;

import mir.experimental.model.rbf;


class ModelNotTrainedException : Exception
{
    ///
    this(
        string msg,
        string file = __FILE__,
        uint line = cast(uint)__LINE__,
        Throwable next = null
        ) pure nothrow @nogc @safe
    {
        super(msg, file, line, next);
    }
}

/++
Radial basis function artificial neural network model.

Single-layered, linear, non-parametric artificial neural network model.

Params:
    T = Value type used, has to be floating point. Default is double.
    rbf = Radial basis function used. Default is gaussian.

See:
    mir.experimental.model.rbf
+/
struct RbfNetwork(T = double, alias rbf = gaussian!T)
    if (isFloatingPoint!T && hasRbfInterface!rbf)
{
private:

    alias Vector = Slice!(1, T*);
    alias Matrix = Slice!(2, T*);

    Matrix _centers; // training centers (input layer of the network).
    Matrix _weights; // calculated training weights (hidden layer of the network).

    T _radius; // radius of a basis function.
    T _lambda; // regularization value for ridge regression.

public:

    /++
    Default constructor of the model.

    Params:
        radius = Radius value of radial basis function, used in both training, and fitting.
        lambda = Regularization parameter.
    +/
    this(T radius, T lambda = 1e-4)
    in
    {
        assert(radius > 0.0, "Radius value has to be larger than 0.");
    }
    body
    {
        this._radius = radius;
        this._lambda = lambda;
    }

    /++
    Returns: radius value used by this model.
    +/
    auto radius() const @property
    {
        return this._radius;
    }

    /++
    Returns: regularization value used by this model.
    +/
    auto lambda() const @property
    {
        return this._lambda;
    }

    /++
    Check if this model has been previously trained.
    +/
    auto isTrained() const @property
    {
        return !_centers.empty;
    }

    /++
     Train neural network with given data.

     Params:
         centers = Center values, data node positions (m-by-n).
         values = Training value, outcome of trained function at center position (m-by-p).
     Returns:
         Trained model.
    +/
    ref train(Matrix centers, Matrix values)
    in
    {
        assert(centers.length == values.length);
    }
    body
    {
        alias x = centers;
        alias y = values;

        _centers = x.slice;
        _weights = slice!T(y.shape);

        auto H = slice!T(x.length, x.length);

        designRbf!rbf(x, x, _radius, H);
        ridgeGlobalWeights(y, H, _lambda, _weights);

        return this;
    }

    /// ditto
    ref train(Vector centers, Vector values)
    {
        return train(centers.reshape(centers.length, 1), values.reshape(values.length, 1));
    }

    /++
     Perform data fitting based on previous neural network training.

     Params:
        queryCenters = Query center values to fit trained model to.

    Returns:
        Resulting fitting values matrix, of same row size as query centers
        and column 
    +/
    Matrix fit(Matrix queryCenters)
    in
    {
        assert(queryCenters.length!1 == _centers.length!1,
                "Query centers' dimensionality is not equal with input centers.");
    }
    body
    {
        if (!isTrained)
        {
            throw new ModelNotTrainedException("Model used for data fitting has not been trained.");
        }

        alias qx = queryCenters;
        alias w = _weights;
        alias x = _centers;

        auto pq = qx.length;
        auto p = w.length!0;
        auto d = w.length!1;
        auto Ht = slice!T(pq, p);
        auto ft = slice!T(pq, d);

        designRbf!rbf(qx, x, _radius, Ht);

        gemm(1.0, Ht, w, 0.0, ft);

        return ft;
    }

    /// ditto
    Vector fit(Vector queryCenters)
    {
        auto ft = fit(queryCenters.reshape(queryCenters.length, 1));
        return ft.reshape(queryCenters.length);
    }

    // void save();
    // void load();
}

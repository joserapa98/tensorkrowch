#include <torch/extension.h>
#include <tuple>
#include <array>
#include <vector>


at::Tensor contract(
    torch::Tensor& tensor1,
    torch::Tensor& tensor2,
    std::tuple<at::IntArrayRef, at::IntArrayRef> permutation_dims,
    std::tuple<int, int, int> shape_limits)
{
    size_t batch = std::get<0>(shape_limits);
    size_t non_contract_0 = std::get<1>(shape_limits);
    size_t contract = std::get<2>(shape_limits);

    size_t size = batch + 2;
    size_t new_size = tensor1.sizes().size() + tensor2.sizes().size()
                        - 2 * contract - batch;

    torch::Tensor* permute1 = &tensor1;
    torch::Tensor* permute2 = &tensor2;

    torch::Tensor aux1;
    torch::Tensor aux2;

    /* Permute if needed */
    if (std::get<0>(permutation_dims).size() > 0)
    {
        aux1 = tensor1.permute(std::get<0>(permutation_dims));
        permute1 = &aux1;
    }

    if (std::get<1>(permutation_dims).size() > 0)
    {
        aux2 = tensor2.permute(std::get<1>(permutation_dims));
        permute2 = &aux2;
    }

    /* Compute sizes for reshapes */
    int64_t* aux_shape1 = new int64_t[size];
    int64_t* aux_shape2 = new int64_t[size];
    int64_t* new_shape = new int64_t[new_size];

    // batch edges
    int64_t aux_size;
    int j = 0;
    for (size_t i = 0; i < batch; i++)
    {
        aux_size = permute1->size(i);
        aux_shape1[i] = aux_size;
        aux_shape2[i] = aux_size;
        new_shape[j] = aux_size;
        j++;
    }

    // non contracted edges node1
    int64_t cum_prod = 1;
    for (size_t i = batch; i < batch + non_contract_0; i++)
    {
        aux_size = permute1->size(i);
        cum_prod *= aux_size;
        new_shape[j] = aux_size;
        j++;
    }
    aux_shape1[size - 2] = cum_prod;

    // contract edges
    cum_prod = 1;
    for (size_t i = batch + non_contract_0; i < tensor1.sizes().size(); i++)
    {
        aux_size = permute1->size(i);
        cum_prod *= aux_size;
    }
    aux_shape1[size - 1] = cum_prod;
    aux_shape2[size - 2] = cum_prod;

    // non contracted edges node2
    cum_prod = 1;
    for (size_t i = batch + contract; i < tensor2.sizes().size(); i++)
    {
        aux_size = permute2->size(i);
        cum_prod *= aux_size;
        new_shape[j] = aux_size;
        j++;
    }
    aux_shape2[size - 1] = cum_prod;

    /* Reshape and contract */
    at::IntArrayRef aux_shape1_ref(aux_shape1, size);
    at::IntArrayRef aux_shape2_ref(aux_shape2, size);
    at::IntArrayRef new_shape_ref(new_shape, new_size);

    auto reshape1 = permute1->reshape(aux_shape1_ref);
    auto reshape2 = permute2->reshape(aux_shape2_ref);

    auto result = reshape1.matmul(reshape2).view(new_shape_ref);

    delete[] aux_shape1;
    delete[] aux_shape2;
    delete[] new_shape;

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("contract", &contract, "Cpp extension for contract_between");
}
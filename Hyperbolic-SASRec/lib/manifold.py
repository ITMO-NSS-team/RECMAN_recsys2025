import torch
import torch.utils.checkpoint as checkpoint





def hyperbolic_dist(emb1, emb2, c=1.0):
    """
    Compute hyperbolic distance in the Poincaré ball model with curvature c.
    emb1, emb2: (batch_size, num_items, emb_dim)
    c: positive float (curvature)
    """
    # Compute norms of embeddings
    u_norm_sq = torch.sum(emb1 ** 2, dim=-1)  # (batch_size, num_items)
    v_norm_sq = torch.sum(emb2 ** 2, dim=-1)  # (batch_size, num_items)
    
    # Compute inner product and then squared Euclidean distance
    inner_prod = torch.sum(emb1 * emb2, dim=-1)  # (batch_size, num_items)
    euclidean_dist_sq = u_norm_sq + v_norm_sq - 2 * inner_prod

    # Incorporate curvature into denominators and numerator
    denom = (1 - c * u_norm_sq) * (1 - c * v_norm_sq) + 1e-5
    x = 1 + 2 * c * euclidean_dist_sq / denom
    x = torch.clamp(x, min=1 + 1e-5)
    
    # Scale the result appropriately
    dist = (2 / torch.sqrt(torch.tensor(c, device=emb1.device))) * torch.acosh(x)
    return dist

def seq_euclidean_dist(eseq1, eseq2, single_dist):
    """
    Compute sequence-wise Euclidean distance using a distance function (batched).
    eseq1, eseq2: (batch_size, num_items, emb_dim)
    """
    dist_vector = single_dist(eseq1, eseq2)  # (batch_size, num_items)
    eseq_dist = torch.norm(dist_vector, dim=-1)  # (batch_size)
    return eseq_dist


def pairseq_dist_affinity(seq, single_dist, chunk_size=64, return_dist=False):
    """
    Compute the pairwise sequence distance affinity matrix in chunks.
    seq: (batch_size, num_items, emb_dim)
    single_dist: function to compute distance between two sequences
    chunk_size: number of sequences to process in one chunk
    """
    batch_size = seq.shape[0]
    affinity = torch.empty(batch_size, batch_size, device=seq.device)
    dist = torch.empty(batch_size, batch_size, device=seq.device)
    
    for i in range(0, batch_size, chunk_size):
        seq_chunk = seq[i:i+chunk_size]  # (chunk_size, num_items, emb_dim)
        # Expand dimensions to compute pairwise distances against the whole batch
        seq_chunk_exp = seq_chunk.unsqueeze(1)   # (chunk_size, 1, num_items, emb_dim)
        seq_all = seq.unsqueeze(0)                 # (1, batch_size, num_items, emb_dim)
        
        # Compute distance: returns (chunk_size, batch_size)
        chunk_dist = seq_euclidean_dist(seq_chunk_exp, seq_all, single_dist)
        dist[i:i+chunk_size] = chunk_dist
        affinity[i:i+chunk_size] = torch.exp(-chunk_dist ** 2)
    
    
    if return_dist:
        return dist,affinity
    else:
        return affinity


def project_embeddings(embeddings, c=1.0, eps=1e-5):
    """
    Project embeddings back into the Poincaré ball of curvature c.
    
    Args:
        embeddings (torch.Tensor): Tensor of shape (..., emb_dim).
        c (float): Curvature parameter. For a Poincaré ball of curvature -c, 
                   valid embeddings satisfy ||x|| < 1/sqrt(c).
        eps (float): Small epsilon to avoid exactly reaching the boundary.
    
    Returns:
        torch.Tensor: The projected embeddings.
    """
    # The radius of the ball is 1/sqrt(c)
    r = 1.0 / torch.sqrt(torch.tensor(c, device=embeddings.device))
    
    norm = embeddings.norm(dim=-1, keepdim=True)
    # Determine scaling factors: if norm > r, scale down to (r - eps)
    scale = (r - eps) / norm
    # Only scale embeddings where norm exceeds (r - eps); otherwise, keep as is.
    scale = torch.where(norm > (r - eps), scale, torch.ones_like(scale))
    return embeddings * scale


def estimate_curvature_SVD(emb, c=1.0):

    emb1 = emb.unsqueeze(1)

    emb2 = emb.unsqueeze(0)

    D = hyperbolic_dist(emb1, emb2, c=c)

    K = torch.exp(-D**2)

    U, S, V = torch.svd(K)

    ratio = S[1] / S[0]

    print("Estimated curvature SVD c =", ratio.item())


#this fellow uses less memory, but makes something wrong with gradient
# def hyperbolic_dist(emb1, emb2):
#     # Wrap the computation in a function suitable for checkpointing.
#     def compute_dist(emb1, emb2):
#         """
#         Compute hyperbolic distance in the Poincaré ball model for batched inputs.
#         emb1, emb2: (batch_size, num_items, emb_dim)
#         """
#         # Compute norms without broadcasting the full difference tensor.
#         u_norm_sq = torch.sum(emb1 ** 2, dim=-1)  # (batch_size, num_items)
#         v_norm_sq = torch.sum(emb2 ** 2, dim=-1)  # (batch_size, num_items)
    
#         # Compute inner product and then Euclidean distance squared
#         inner_prod = torch.sum(emb1 * emb2, dim=-1)  # (batch_size, num_items)
#         euclidean_dist_sq = u_norm_sq + v_norm_sq - 2 * inner_prod

#         # Denom and hyperbolic distance as before.
#         denom = (1 - u_norm_sq) * (1 - v_norm_sq) + 1e-5
#         x = 1 + 2 * euclidean_dist_sq / denom
#         x = torch.clamp(x, min=1 + 1e-5)
#         return torch.acosh(x)
    
#     # Use checkpointing to save memory.
#     return checkpoint.checkpoint(compute_dist, emb1, emb2, use_reentrant=False)


#this fellow is just for reference, better use SVD than direct definition
# class CurvatureEstimator:
#     def __init__(self, dist_fn, num_samples=10000, epsilon=1e-5):
#         self.dist_fn = dist_fn
#         self.num_samples = num_samples
#         self.epsilon = epsilon
#         self.delta_p = torch.log1p(torch.sqrt(torch.tensor(2.0)))  # ln(1 + sqrt(2))

#     def sample_quadruples(self, emb):
#         N = emb.size(0)
#         idx = torch.randint(0, N, (self.num_samples, 4), device=emb.device)
#         return idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]

#     def gromov_product(self, d_xy, d_xz, d_yz):
#         return 0.5 * (d_xz + d_yz - d_xy)

#     def estimate_delta(self, emb):
#         x, y, z, w = self.sample_quadruples(emb)
#         d = lambda a, b: self.dist_fn(emb[a], emb[b])

#         gp1 = self.gromov_product(d(x, y), d(x, w), d(y, w))
#         gp2 = self.gromov_product(d(y, z), d(y, x), d(z, x))
#         gp3 = self.gromov_product(d(z, x), d(z, y), d(x, y))

#         delta = torch.min(torch.stack([gp1, gp2, gp3]), dim=0).values
#         return delta.max()

#     def estimate_diameter(self, emb):
#         # estimate diameter via random samples
#         idx_a = torch.randint(0, emb.size(0), (self.num_samples,), device=emb.device)
#         idx_b = torch.randint(0, emb.size(0), (self.num_samples,), device=emb.device)
#         d_ab = self.dist_fn(emb[idx_a], emb[idx_b])
#         return torch.quantile(d_ab, 0.95)  # robust diameter

#     def curvature(self, emb):
#         delta = self.estimate_delta(emb)
#         diam = self.estimate_diameter(emb)
#         delta_rel = 2 * delta / diam

#         diam_P = 2 * torch.log(torch.Tensor([(1 + 2 * (1 - self.epsilon)) / self.epsilon])).to(delta_rel.device)
#         delta_rel_p = 2 * self.delta_p / diam_P

#         c_est = (delta_rel_p / delta_rel) ** 2
#         return c_est.item()

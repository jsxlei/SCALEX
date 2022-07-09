suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(scater))
suppressPackageStartupMessages(library(future))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(future.apply))

parser <- ArgumentParser(description='Seurat_v3 for the integrative analysis of multi-batch single-cell transcriptomic profiles')

parser$add_argument("-i", "--input_path", type="character", help="Path contains RNA data")
parser$add_argument("-o", "--output_path", type="character", default='./', help="Output path")
parser$add_argument("-mf", "--minFeatures", type="integer", default=600, help="Remove cells with less than minFeatures features")
parser$add_argument("-mc", "--minCells", type="integer", default=3, help="Remove features with less than minCells cells")
parser$add_argument("-nt", "--n_top_features", type="integer", default=2000, help="N highly variable features")
args <- parser$parse_args()

plan("multiprocess", workers = 1)
options(future.globals.maxSize = 100000 * 1024^2)

message('Reading matrix.mtx and metadata.txt in R...')
data <- readMM(paste(args$input_path, '/matrix.mtx', sep=''))
genes <- read.table(paste(args$input_path, '/genes.txt', sep=''), sep='\t')
metadata <- read.csv(paste(args$input_path, '/metadata.txt', sep=''), sep='\t')
row.names(metadata) <- metadata[,1]
metadata <- metadata[,-1]
metadata$batch = as.character(metadata$batch)
# metadata$celltype = as.character(metadata$celltype)

data <- t(data)

colnames(data) <- row.names(metadata) 
row.names(data) <- genes[,1]

adata <- CreateSeuratObject(as(data, "sparseMatrix"), 
                           meta.data = metadata,
                           min.cells = args$minCells, 
                           min.features = args$minFeatures)


message('Preprocessing...')
adata.list <- SplitObject(adata, split.by = "batch")

print(dim(adata)[2])
if(dim(adata)[2] < 50000){
    for (i in 1:length(adata.list)) {
        adata.list[[i]] <- NormalizeData(adata.list[[i]], verbose = FALSE)
        adata.list[[i]] <- FindVariableFeatures(adata.list[[i]], selection.method = "vst", nfeatures = args$n_top_features, verbose = FALSE)
    }
    message('FindIntegrationAnchors...')
    adata.anchors <- FindIntegrationAnchors(object.list = adata.list, dims = 1:30,verbose =FALSE,k.filter = 30)
#     adata.anchors <- FindIntegrationAnchors(object.list = adata.list, dims = 1:30,verbose =FALSE,k.filter = 100)

    message('IntegrateData...')
    adata.integrated <- IntegrateData(anchorset = adata.anchors, dims = 1:30, verbose = FALSE)
}else{
    adata.list <- future_lapply(X = adata.list, FUN = function(x) {
    x <- NormalizeData(x, verbose = FALSE)
    x <- FindVariableFeatures(x, nfeatures = args$n_top_features, verbose = FALSE)
    })
    
    features <- SelectIntegrationFeatures(object.list = adata.list)
    adata.list <- future_lapply(X = adata.list, FUN = function(x) {
        x <- ScaleData(x, features = features, verbose = FALSE)
        x <- RunPCA(x, features = features, verbose = FALSE)
    })
    message('FindIntegrationAnchors...')
    adata.anchors <- FindIntegrationAnchors(object.list = adata.list, dims = 1:30, verbose =FALSE, reduction = 'rpca', reference = c(1, 2))
    message('IntegrateData...')
    adata.integrated <- IntegrateData(anchorset = adata.anchors, dims = 1:30, verbose = FALSE)
}

if (!file.exists(args$output_path)){
    dir.create(file.path(args$output_path),recursive = TRUE)
}

message('Writing...')
write.table(adata.integrated@meta.data[,-1], paste(args$output_path, "/metadata.txt", sep=''), sep='\t')
write.table(t(data.frame(adata.integrated@assays$integrated@data)), paste(args$output_path, "/integrated.txt", sep=''), sep='\t')
rm(list = ls())
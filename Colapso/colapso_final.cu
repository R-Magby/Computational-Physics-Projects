#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h> 
#include <string.h>

    //ecu:

        //dr_dr_R= 1/(2*R) - 0.25*R ( 2*K* ( PI*PI-L ))
        //dt_phi=PI
        //dt_d_phi=d_PI
        //dt_PI=d_rrs/(R*R)
        //pongamo dR=0
    //minsko
    //int alpha=1;
    //int beta=0;
    //con esto se desprende que  K_r^r=0, Lx=1,Lxx=0 
//crack nicholson (interacion)


__global__ void evolution_chi(float *P,float *S , float *chi, float *N, float *beta,float dt,int t,int Nx, int Nt){
    int x =threadIdx.x + blockDim.x*blockIdx.x;

    if(t*Nx+x < Nx*Nt-1 && x<Nx){
        chi[(t+1)*Nx+x]=(N[t*Nx+x]*P[x] + beta[x]*S[x])*dt + chi[(t)*Nx+x];
}
/*
if (x==20*10000/200 && t<Nt){
    printf("En x=20 en t=%d, chi=%f, N=%f, S=%f, P=%f, beta=%f \n",t,chi[(t+1)*Nx+x],N[t*Nx+x],S[x],P[x],beta[x]);
}
if (x==0 && t<Nt){
    printf("En x=Nx-1 en t=%d, chi1=%f, N=%f, S=%f, P=%f, beta=%f \n",t,chi[(t+1)*Nx+x],N[t*Nx+x],S[x],P[x],beta[x]);
}
if (x==Nx-2 && t<Nt){
    printf("En x=Nx-1 en t=%d, chi=%f, N=%f, S=%f, P=%f, beta=%f \n",t,chi[(t+1)*Nx+x],N[t*Nx+x],S[x],P[x],beta[x]);
}*/
}

__global__ void evolution_S(float *f ,float *P,float *snake_u_1,float *beta, float *N, float *K_r, float dr,float dt,int Nx, int t, int interaciones){

    int x =threadIdx.x + blockDim.x*blockIdx.x;
 
    float df,dN,dP;
    float avg_u;
    for (int j=1;j<interaciones+1;j++){
        if(x<Nx){
            snake_u_1[x]=f[x];

        }
    if(x<Nx && x>=0){
        if (x==Nx-1){
            df=(snake_u_1[x]-snake_u_1[x-1])/(dr);
            dN=(N[t*Nx+x]-N[t*Nx+x-1])/(dr);
        
            dP=(P[x]-P[x-1])/(dr);

            avg_u = (beta[x]*df + P[x]*dN+  N[t*Nx+x]*(dP+K_r[x]*snake_u_1[x]))*dt+f[x];
            snake_u_1[x]= 0.5*(avg_u +f[x]);
        }
        else if (x==0){
            df=(snake_u_1[x+1]-snake_u_1[x])/(dr);
            dN=0.0;
            
            dP=(P[x+1]-P[x])/(dr);
    
            avg_u = (beta[x]*df + P[x]*dN+  N[t*Nx+x]*(dP+K_r[x]*snake_u_1[x]))*dt+f[x];
            snake_u_1[x]= 0.5*(avg_u +f[x]);
        }
        else{
            df=(snake_u_1[x+1]-snake_u_1[x-1])/(2.0*dr);
            dN=(N[t*Nx+x+1]-N[t*Nx+x-1])/(2.0*dr);
            
            dP=(P[x+1]-P[x-1])/(2.0*dr);
    
            avg_u = (beta[x]*df + P[x]*dN+  N[t*Nx+x]*(dP+K_r[x]*snake_u_1[x]))*dt+f[x];
            snake_u_1[x]= 0.5*(avg_u +f[x]);
        }

    }
    __syncthreads();
    }
    if(x<Nx && x>=0){
        if (x==Nx-1){
            df=(snake_u_1[x]-snake_u_1[x-1])/(dr);
            dN=(N[t*Nx+x]-N[t*Nx+x-1])/(dr);
        
            dP=(P[x]-P[x-1])/(dr);

            f[x] = (beta[x]*df + P[x]*dN+  N[t*Nx+x]*(dP+K_r[x]*snake_u_1[x]))*dt+f[x];
        }
        else if (x==0){
            df=(snake_u_1[x+1]-snake_u_1[x])/(dr);
            dN=0.0;
        
            dP=(P[x+1]-P[x])/(dr);

            f[x] = (beta[x]*df + P[x]*dN+  N[t*Nx+x]*(dP+K_r[x]*snake_u_1[x]))*dt+f[x];
        }
        else{
            df=(snake_u_1[x+1]-snake_u_1[x-1])/(2.0*dr);
            dN=(N[t*Nx+x+1]-N[t*Nx+x-1])/(2.0*dr);
            
            dP=(P[x+1]-P[x-1])/(2.0*dr);
    
            f[x] = (beta[x]*df + P[x]*dN+  N[t*Nx+x]*(dP+K_r[x]*snake_u_1[x]))*dt+f[x];
        }
    }

    __syncthreads();

}
__global__ void evolution_P(float *f , float *snake_u_1, float *S, float *beta, float *N, float *R, float dr,float dt,int Nx,int t, int interaciones){

    int x =threadIdx.x + blockDim.x*blockIdx.x;
 
    float RS_m1,RS_p1;
    float df,dN,dRS;
    float avg_u;
    float alpha;
    alpha=0.01;
    for (int j=1;j<interaciones+1;j++){
        if(x<Nx){
            snake_u_1[x]=f[x];

        }
    if(x<Nx && x>=0){
        if (x==Nx-1){

        df=(snake_u_1[x]-snake_u_1[x-1])/(dr);
        dN=(N[t*Nx+x]-N[t*Nx+x-1])/(dr);

        RS_m1=R[x-1]*R[x-1]*S[x-1];
        RS_p1=R[x]*R[x]*S[x];

        dRS=(RS_p1 - RS_m1)/(dr);

        avg_u = ( beta[x]*df + S[x]*dN+  N[t*Nx+x]*dRS/(R[x]*R[x]+alpha))*dt+f[x];
        snake_u_1[x]=0.5*(avg_u+f[x]);
        }
        else if(x==0){
            df=(snake_u_1[x+1]-snake_u_1[x])/(dr);
            dN=0.0;
    
            RS_m1=R[x]*R[x]*S[x];
            RS_p1=R[x+1]*R[x+1]*S[x+1];
    
            dRS=(RS_p1 - RS_m1)/(dr);
    
            avg_u = ( beta[x]*df + S[x]*dN+  N[t*Nx+x]*dRS/(R[x]*R[x]+alpha))*dt+f[x];
            snake_u_1[x]=0.5*(avg_u+f[x]);
        }
        else{    
            df=(snake_u_1[x+1]-snake_u_1[x-1])/(2.0*dr);
            dN=(N[t*Nx+x+1]-N[t*Nx+x-1])/(2.0*dr);
    
            RS_m1=R[x-1]*R[x-1]*S[x-1];
            RS_p1=R[x+1]*R[x+1]*S[x+1];
    
            dRS=(RS_p1 - RS_m1)/(2.0*dr);
    
            avg_u = ( beta[x]*df + S[x]*dN+  N[t*Nx+x]*dRS/(R[x]*R[x]+alpha))*dt+f[x];
            snake_u_1[x]=0.5*(avg_u+f[x]);
        }
       // Pi[x] = 0.5*(temp2-temp1)*dt/dr  +  Pi[x];
    }
    __syncthreads();

}

    if(x<Nx && x>=0){
        if (x==Nx-1){
            df=(snake_u_1[x]-snake_u_1[x-1]) / (dr);
            dN=(N[t*Nx+x]-N[t*Nx+x-1])/(dr);
    
            RS_m1=R[x-1]*R[x-1]*S[x-1];
            RS_p1=R[x]*R[x]*S[x];
    
            dRS=(RS_p1 - RS_m1)/(dr);
    
            f[x] = ( beta[x]*df + S[x]*dN+  N[t*Nx+x]*dRS/(R[x]*R[x]+alpha))*dt+f[x];
        }
        else if(x==0){    
            df=(snake_u_1[x+1]-snake_u_1[x]) / (dr);
            dN=0.0;

            RS_m1=R[x]*R[x]*S[x];
            RS_p1=R[x+1]*R[x+1]*S[x+1];

            dRS=(RS_p1 - RS_m1)/(dr);

            f[x] = ( beta[x]*df + S[x]*dN+  N[t*Nx+x]*dRS/(R[x]*R[x]+alpha))*dt+f[x];  
        }  

        else{

            df=(snake_u_1[x+1]-snake_u_1[x-1]) / (2.0*dr);
            dN=(N[t*Nx+x+1]-N[t*Nx+x-1])/(2.0*dr);

            RS_m1=R[x-1]*R[x-1]*S[x-1];
            RS_p1=R[x+1]*R[x+1]*S[x+1];

            dRS=(RS_p1 - RS_m1)/(2*dr);

            f[x] = ( beta[x]*df + S[x]*dN+  N[t*Nx+x]*dRS/(R[x]*R[x]+alpha))*dt+f[x];
        }

       // Pi[x] = 0.5*(temp2-temp1)*dt/dr  +  Pi[x];
    }

    __syncthreads();

}
__global__ void evolution_R(float *f , float *snake_u_1,  float *beta, float *N, float *K_r, float dr,float dt,int Nx,int t, int interaciones){
    int x =threadIdx.x + blockDim.x*blockIdx.x;

    float df;
    float avg_u;
    for (int j=1;j<interaciones+1;j++){
        if(x<Nx){
            snake_u_1[x]=f[x];

        }
    if(x<Nx && x>=0){
        if (x==Nx-1){
            df=(snake_u_1[x]-snake_u_1[x-1])/(dr);

            avg_u = ( beta[x]*df + N[t*Nx+x]*0.5*snake_u_1[x]*K_r[x])*dt+f[x];
            snake_u_1[x]=0.5*(avg_u+f[x]);
        }
        else if (x==0){
            df=1.0;

            avg_u = ( beta[x]*df + N[t*Nx+x]*0.5*snake_u_1[x]*K_r[x])*dt+f[x];
            snake_u_1[x]=0.5*(avg_u+f[x]);
        }

        else{
            df=(snake_u_1[x+1]-snake_u_1[x-1])/(2*dr);

            avg_u = ( beta[x]*df + N[t*Nx+x]*0.5*snake_u_1[x]*K_r[x])*dt+f[x];
            snake_u_1[x]=0.5*(avg_u+f[x]);
        }
    }
    __syncthreads();
    }
    if(x<Nx && x>=0){
        if (x==Nx-1){
            df=(snake_u_1[x]-snake_u_1[x-1])/(dr);

            f[x] = ( beta[x]*df + N[t*Nx+x]*0.5*snake_u_1[x]*K_r[x])*dt+f[x];
        }
        else if (x==0){
            df=1.0;

            f[x] = ( beta[x]*df + N[t*Nx+x]*0.5*snake_u_1[x]*K_r[x])*dt+f[x];
        }
        else{
            df=(snake_u_1[x+1]-snake_u_1[x-1])/(2.0*dr);

            f[x] = ( beta[x]*df + N[t*Nx+x]*0.5*snake_u_1[x]*K_r[x])*dt+f[x];
        }
    }
    __syncthreads();
    /*if (x==0){
        printf(" R_en 0=%f,\n",f[x]);
    }
    if (x==210){
        printf(" R=%f,\n",f[x]);
    }
    if (x==Nx-1){
        printf(" R_final=%f,\n",f[x]);
    }
    if (x==Nx-2){
        printf(" R_final2=%f,\n",f[x]);
    }
*/
}

__global__ void N_lapso(float *N, float *R,float *K_r , float *P,float dr, int Nx, int Nt, float *a, float *b, float *c, float *c_prima, int t){
    int x =((threadIdx.x + blockDim.x*blockIdx.x));
    float k=3.1415*8.0;
    float T;
    float dR;
    float alpha;
    if (x<Nx){
        if (x==0){
            b[x]= -1.0/dr;
            c[x]= 1.0/dr;
        }
        else if (x==Nx-1){
            b[x]=1.0;
            a[x]=0.0;
        }
        else{
            dR=(R[x+1]-R[x-1])/(2.0*dr);
            T=(1.5*K_r[x]*K_r[x] + k*(P[x]*P[x]));

            a[x] = (R[x] - dr*dR);
            b[x] = -(2.0*R[x] + T*dr*dr*R[x]);
            c[x] = (R[x] + dr*dR);
        }
    }
    __syncthreads();
    if (x==0){

        for(int j=0;j<Nx-1;j++){
            if (j==0){
                c_prima[j] = c[j] / b[j];
            }

            else {
                c_prima[j] = c[j] / ( b[j] - a[ j ] * c_prima[j-1] );

            }
        }
    
        for(int j=Nx-1;j>=0;j--){

            if (j==Nx-1){
                N[ t*Nx + j ] = 1.0 / ( b[j] - a[ j ] * c_prima[j-1]);
            }
            else{
                N[ t*Nx + j ] = 0.0 - c_prima[j] * N[ t*Nx + j  + 1];
            }

        }       
    }
    __syncthreads();
}
__global__ void beta_constrain(float *beta,float *K_r, float *N,float dr, int Nx ,  int t){
    int x =threadIdx.x + blockDim.x*blockIdx.x;
    float kn,sum_par,sum_impar;
    sum_par=0.0;
    sum_impar=0.0;

        if(x<Nx && x>=0){
            kn=N[t*Nx+x]*K_r[x];

            if (x==0){
                beta[x]=N[t*Nx+x]*K_r[x];
            }
            else if(x>0){
                for(int i=2; i<x; i=2+i){
                    sum_par+=N[t*Nx+i]*K_r[i];
                }
                for(int i=1; i<x; i=2+i){
                    sum_impar+=N[t*Nx+i]*K_r[i];
                }
                beta[x] = (beta[0] +4.0*sum_impar + 2.0*sum_par +  kn)*(dr/3.0);
            }
        }

    __syncthreads();

}
__global__ void KA(float *k_r,float *S , float *P,float *R,float dr, int Nx){
    int x =threadIdx.x + blockDim.x*blockIdx.x;
    float k=3.1415*8.0;
    float kn,sum_par,sum_impar;
    float alpha;
    sum_par=0.0;
    sum_impar=0.0;
    alpha=0.01;

        if(x<Nx ){
            kn=P[x]*S[x]*R[x]*R[x]*R[x];

            if (x==0){
                k_r[x]=-k*P[x]*S[x]*R[x]*R[x]*R[x]*(dr/3.0)/(R[x]*R[x]*R[x]+alpha);
            }
            else if(x>0){
                for(int i=2; i<x; i=2+i){
                    sum_par+=P[i]*S[i]*R[i]*R[i]*R[i];
                }
                for(int i=1; i<x; i=2+i){
                    sum_impar+=P[i]*S[i]*R[i]*R[i]*R[i];
                }
                k_r[x] = -k*(k_r[0] +4.0*sum_impar + 2.0*sum_par +  kn)*(dr/3.0)/(R[x]*R[x]*R[x]+alpha);
            }
        }
        /*
        if (x==210){
            printf(" K=%f,\n",k_r[x]);
        }
        if (x==Nx-1){
            printf(" K_final=%f,\n",k_r[x]);
                }
        if (x==0){
            printf(" K_final2=%f,\n",k_r[x]);
        }*/
    __syncthreads();

}

void guardar_salida_chi(float *data,int Nr, int T) {
    FILE *fp = fopen("campo_escalar_colapso.dat", "wb");
    fwrite(data, sizeof(float), Nr*T, fp);
    fclose(fp);
}

void guardar_salida_N(float *data,int Nr, int T) {
    FILE *fp = fopen("lapso_colapso.dat", "wb");
    fwrite(data, sizeof(float), Nr*T, fp);
    fclose(fp);
}

void cargar_R(float *data, int Nr){

    FILE *arch;
    arch=fopen("test.npy","rb");
    if (arch==NULL)
        exit(1);
    fread( data , sizeof(float) , Nr , arch );
    fclose(arch);

}

void rellenar_chi(float *chi, float *r, int size, int n, float A){
    float r_0=20.0;
    float sigma2=1.5*1.5;

    if (n==1){
        for (int i=0;i<size;i++){
            chi[i]=-2.0*A*(r[i]-r_0)/sigma2 * exp(-(r[i]-r_0)*(r[i]-r_0)/sigma2);
        }
    }

    else{
        for (int i=0;i<size;i++){
            chi[i]=1.0;
        }
    }

}

void rellenar_r(float *r, int N, float dr){
    for(int i=0;i<N;i++){
        r[i]=dr*i;
    }
}

void rellenar(float *r, int N, float dr ,float num){
    for(int i=0;i<N;i++){
        r[i]=num;
    }
}

void rellenar_dr_chi(float *phi_n, float *chi, int N,float dr){
    phi_n[0]=(chi[1]-chi[0])/(dr);
    for(int i=1;i<N-1;i++){
        phi_n[i] = (chi[i+1]-chi[i-1])/(2.0*dr) ;    
    }
    phi_n[N-1]=(chi[N-1]-chi[N-2])/(dr);

}

void rellenar_R_1(float *R, float *K_r,float *S,int Nx,float dr){

    float dR,error_1,error_2, temp;
    float k=3.1415*8.0;
    float alpha=0.01;
    
    float epsilon = 1e-6;

    int i=0;
    error_1=1.0;

    while ( error_1 > 0.0001 ){
    R[0]=0.0;
    R[1]=dr;

    error_2=0.0;

        for (int x=1;x<Nx-1;x++){ 

                if(x==0){
                    temp=(R[x+1]-R[x])/(dr);
                    if (temp < epsilon) {
                        temp = epsilon;
                    }
                    error_2+= (temp-1.0)*(temp-1.0)*0.5 ;
                }
                else if(x>0){
                    dR=(R[x+1]-R[x-1])/(2.0*dr);
                    temp=-(-(1.0-dR*dR)*dr*dr/(  (R[x+1]+R[x-1])/(2.0)  +alpha) -R[x+1]-R[x-1])/(2.0 + 0.25* (R[x+1]+R[x-1])/(2.0) *(k*(S[x]*S[x]))*dr*dr);

                    //temp=-(1.0-dR*dR)*dr*dr/(R[x]+alpha)+0.25*R[x]*(2.0*k*(S[x]*S[x]))*dr*dr+2.0*R[x]-R[x-1];
                    if (temp < epsilon) {
                        temp = epsilon;
                    }
                    error_2+= (temp-R[x+1])*(temp-R[x+1])*0.5 ;
                }

                R[x+1]=temp;
                if (isnan(temp)) {
                    printf("NaN detected at x = %d\n", x);
                    printf("dR = %f, R[%d] = %f, R[%d] = %f, R[%d] = %f\n", dR, x, R[x], x - 1, R[x - 1], x + 1, R[x + 1]);
                    printf("S[%d] = %f, temp = %f\n", x, S[x], temp);
                    return;
                }
        }


        
    printf("error= %f\n",error_2);
    error_1=error_2;
    i++;

    }

}

int main(int argc, char *argv[]){
    printf("Iniciando final8.cu...\n");
    clock_t tiempo_inicio, tiempo_final,tiempo_inicio_gpu,tiempo_final_gpu;
    double segundos,segundos_gpu;
    
    tiempo_inicio = clock();

    float *P, *S , *chi, *beta, *K_r, *N,*R, *r;

    float *cuda_P, *cuda_S , *cuda_chi, *cuda_beta,*cuda_N, *cuda_K_r, *cuda_R;
    float *cuda_snake_P,*cuda_snake_S,*cuda_snake_R;
    float *a , *b, *c , *c_prima;

    int size_r = 20000;
    int size_time=1000;
    float emax=0.0001;

    //el dt debe ser menor que el dr si no colapsa
    float dr=200.0/size_r;
    float dt=50.0/size_time;

    P=(float *)malloc(size_r*sizeof(float));
    S=(float *)malloc(size_r*sizeof(float));
    beta=(float *)malloc(size_r*sizeof(float));
    K_r=(float *)malloc(size_r*sizeof(float));
    R=(float *)malloc(size_r*sizeof(float));
    N=(float *)malloc(size_r*size_time*sizeof(float));

    chi=(float *)malloc(size_r*size_time*sizeof(float));
    r=(float *)malloc(size_r*sizeof(float));

    printf("Rellenar ...\n");

    rellenar_r(r, size_r,dr);
    rellenar_chi(chi, r, size_r, 1, 0.01);
    rellenar_dr_chi(S,chi,size_r,dr);

    // R en t=0 R=0 en el origen y su derivada en t=0 dR=1 en el origen.
    printf("datos iniciales ...\n");

    rellenar(R,size_r,dr, 0.0);
    rellenar(K_r,size_r,dr,0.0);
    rellenar(P,size_r,dr,0.0);
    rellenar(N,size_r*size_time,dr,0.0);
    rellenar(beta,size_r,dr,0.0);

    tiempo_inicio = clock();
    //rellenar_R_1(R,K_r,S,size_r,dr);
    cargar_R(R,size_r);
    
    tiempo_final = clock();
    segundos = (double)(-tiempo_inicio + tiempo_final) / CLOCKS_PER_SEC; 
    printf("Tiempo de ejecucion en el CPU para el R incial ecu(17):%f\n",segundos);

    
    printf("Creando los cudaMallocs ...\n");
    cudaMalloc((void **)&cuda_P, size_r*sizeof(float)) ;
    cudaMalloc((void **)&cuda_S, size_r*sizeof(float));
    cudaMalloc((void **)&cuda_chi, size_r*size_time*sizeof(float));
    cudaMalloc((void **)&cuda_beta, size_r*sizeof(float)) ;
    cudaMalloc((void **)&cuda_R, size_r*sizeof(float)) ;
    cudaMalloc((void **)&cuda_K_r, size_r*sizeof(float)) ;
    cudaMalloc((void **)&cuda_N, size_r*size_time*sizeof(float)) ;
    cudaMalloc((void **)&cuda_snake_P, size_r*sizeof(float));
    cudaMalloc((void **)&cuda_snake_S, size_r*sizeof(float));
    cudaMalloc((void **)&cuda_snake_R, size_r*sizeof(float));


    cudaMalloc((void **)&a, size_r*sizeof(float));
    cudaMalloc((void **)&b, size_r*sizeof(float));
    cudaMalloc((void **)&c, size_r*sizeof(float));
    cudaMalloc((void **)&c_prima, size_r*sizeof(float));

    printf("Copiando los datos del host al device ...\n");

    cudaMemcpy(cuda_chi, chi, size_r*size_time*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_S, S, size_r*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_K_r, K_r, size_r*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_N, N, size_r*size_time*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_R, R, size_r*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_P, P, size_r*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_beta, beta, size_r*sizeof(float), cudaMemcpyHostToDevice);

    printf("Declarando los threads ...\n");

    int thread=1024;
    dim3 bloque(thread);
    dim3 grid((int)ceil((float)(size_r)/thread));

    tiempo_inicio = clock();


    printf("Ejecutar device:\n");

    int interaciones=2;//clarck Nicholson

    N_lapso<<<grid,bloque>>>(cuda_N,cuda_R, cuda_K_r,cuda_P, dr ,size_r ,size_time ,a ,b ,c ,c_prima ,0);

    cudaDeviceSynchronize();

    for (int t=0;t<size_time-1;t++){
        tiempo_inicio_gpu = clock();

        evolution_chi<<<grid,bloque>>>(cuda_P,cuda_S,cuda_chi,cuda_N,cuda_beta ,dt,t,size_r,size_time);
        cudaDeviceSynchronize();

            evolution_R<<<grid,bloque>>>(cuda_R,cuda_snake_R ,cuda_beta, cuda_N, cuda_K_r,dr ,dt,size_r,t,interaciones);
            cudaDeviceSynchronize();

            evolution_P<<<grid,bloque>>>(cuda_P,cuda_snake_P, cuda_S ,cuda_beta , cuda_N, cuda_R,dr ,dt,size_r,t,interaciones);
            cudaDeviceSynchronize();
            evolution_S<<<grid,bloque>>>(cuda_S,cuda_P,cuda_snake_S, cuda_beta, cuda_N, cuda_K_r,dr ,dt,size_r,t,interaciones);
            cudaDeviceSynchronize();

        KA<<<grid,bloque>>>(cuda_K_r,cuda_S,cuda_P,cuda_R,dr ,size_r);
        cudaDeviceSynchronize();

        N_lapso<<<grid,bloque>>>(cuda_N,cuda_R, cuda_K_r,cuda_P, dr ,size_r,size_time ,a ,b ,c ,c_prima, t+1);
        cudaDeviceSynchronize();

        beta_constrain<<<grid,bloque>>>(cuda_beta, cuda_K_r, cuda_N, dr ,size_r,t+1);
        cudaDeviceSynchronize();

        tiempo_final_gpu = clock();
        if(t%100==0){
            printf("t=%d\n",t);
            segundos_gpu = (double)(-tiempo_inicio_gpu + tiempo_final_gpu) / CLOCKS_PER_SEC; 
            printf("Tiempo de ejecucion del GPU:%f\n",segundos_gpu);
        }
    }

    printf("Ejecucion terminada:\n");
    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n",cudaGetErrorString(err));
    cudaMemcpy(&chi[size_r],&cuda_chi[size_r], (size_time-1)*size_r*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(N,cuda_N, (size_time)*size_r*sizeof(float), cudaMemcpyDeviceToHost);

    //guardar_salida_chi(chi,size_r,size_time);
    //guardar_salida_N(N,size_r,size_time);



tiempo_final = clock();

segundos = (double)(-tiempo_inicio + tiempo_final) / CLOCKS_PER_SEC; 
printf("Tiempo total de ejecucion del GPU:%f\n",segundos);
    cudaFree(cuda_chi);cudaFree(cuda_P);cudaFree(cuda_S);cudaFree(cuda_R);cudaFree(cuda_beta);cudaFree(cuda_N);
    cudaFree(cuda_K_r);cudaFree(cuda_snake_P);cudaFree(cuda_snake_R);cudaFree(cuda_snake_S);
}

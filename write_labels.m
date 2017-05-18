function write_labels(labels)
    [d, n] = size(labels);
    h = fopen('labels.txt', 'wt');
    for i = 1:n
        fprintf(h, '%d\n',labels(1,i));	
    end
    fclose(h);
end
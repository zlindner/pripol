import React, { useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import Search from './search';
import Loader from './loader';

const Container = styled.div`
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
`;

const Name = styled.a`
    position: absolute;
    top: 40px;
    left: 40px;
    font-size: 18px;
    font-weight: 700;
`;

const Error = styled.span`
    position: absolute;
    margin-top: 350px;
    color: #909;
    font-size: 16px;
    user-select: none;
`;

type Props = {
    showAnalysis: Function;
};

const Landing = (props: Props) => {
    // user-entered policy url
    const [url, setURL] = useState('');

    // states
    const [loading, setLoading] = useState(false);
    const [analyzing, setAnalyzing] = useState(false);
    const [error, setError] = useState('');

    const onLoad = () => {
        if (url.length <= 0) {
            setError('Invalid privacy policy url');
            return;
        }

        setError('');
        setLoading(true);

        axios
            .post('/policy/load', { url })
            .then((res) => {
                setLoading(false);
                onAnalyze(res.data.policy);
            })
            .catch((err) => {
                setLoading(false);
                setError('An error occurred while loading privacy policy');
            });
    };

    const onAnalyze = (policy: string[]) => {
        setAnalyzing(true);

        axios
            .post('/model/predict', { policy })
            .then((res) => {
                console.log(res);
                setAnalyzing(false);

                props.showAnalysis(res.data);
            })
            .catch((err) => {
                setAnalyzing(false);
                setError('An error occurred while analyzing privacy policy');
            });
    };

    const hasError = error.length > 0;

    return (
        <Container>
            <Name href='/'>pripol.</Name>

            <Search setURL={setURL} onLoad={onLoad} />

            {hasError && <Error>{error}</Error>}
            {!hasError && (loading || analyzing) && <Loader loading={loading} analyzing={analyzing} url={url} />}
        </Container>
    );
};

export default Landing;

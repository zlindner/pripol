import React from 'react';
import styled from 'styled-components';
import SyncLoader from 'react-spinners/SyncLoader';

const Container = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    position: absolute;
    margin-top: 350px;

    & span {
        padding-top: 20px;
    }
`;

type Props = {
    loading: boolean;
    analyzing: boolean;
    url: string;
};

const Loader = (props: Props) => {
    const matches = props.url.match(/^https?\:\/\/([^\/?#]+)(?:[\/?#]|$)/i);
    const domain = matches && matches[1]; // domain will be null if no match is found

    return (
        <Container>
            <SyncLoader size={10} />

            <span>
                {props.loading && 'Loading'}
                {props.analyzing && 'Analyzing'} privacy policy from {domain}
            </span>
        </Container>
    );
};

export default Loader;
